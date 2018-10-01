package main

import (
	"errors"
	"fmt"
	"math"
	"sync"

	"gocv.io/x/gocv"
)

type Cld struct {
	originalImg gocv.Mat
	result      gocv.Mat
	dog         gocv.Mat
	fDog        gocv.Mat
	etf         *Etf
	wg          sync.WaitGroup
}

type position struct {
	x, y float64
}

const (
	SigmaRatio float64 = 1.6
	SigmaM     float64 = 3.0
	SigmaC     float64 = 1.0
	Rho        float64 = 0.997
	Tau        float64 = 0.8
)

func NewCLD(row, col int) *Cld {
	var wg sync.WaitGroup
	originalImg := gocv.NewMatWithSize(row, col, gocv.MatTypeCV8UC1)
	result := gocv.NewMatWithSize(row, col, gocv.MatTypeCV8UC1)
	dog := gocv.NewMatWithSize(row, col, gocv.MatTypeCV32F)
	fDog := gocv.NewMatWithSize(row, col, gocv.MatTypeCV32F)

	etf := NewETF()
	etf.Init(row, col)

	return &Cld{originalImg, result, dog, fDog, etf, wg}
}

func (c *Cld) Init(row, col int) {
	c.originalImg = gocv.NewMatWithSize(row, col, gocv.MatTypeCV8UC1)
	c.result = gocv.NewMatWithSize(row, col, gocv.MatTypeCV8UC1)
	c.dog = gocv.NewMatWithSize(row, col, gocv.MatTypeCV32F)
	c.fDog = gocv.NewMatWithSize(row, col, gocv.MatTypeCV32F)

	c.etf = NewETF()
	c.etf.Init(row, col)
}

func (c *Cld) ReadSource(file string) error {
	c.originalImg = gocv.IMRead(file, gocv.IMReadGrayScale)
	c.result = gocv.NewMatWithSize(c.originalImg.Rows(), c.originalImg.Cols(), gocv.MatTypeCV8UC1)
	c.dog = gocv.NewMatWithSize(c.originalImg.Rows(), c.originalImg.Cols(), gocv.MatTypeCV8UC1)
	c.fDog = gocv.NewMatWithSize(c.originalImg.Rows(), c.originalImg.Cols(), gocv.MatTypeCV8UC1)

	if err := c.etf.InitEtf(file, c.originalImg); err != nil {
		return errors.New(fmt.Sprintf("Unable to initialize edge tangent flow matrix: %s\n", err))
	}
	return nil
}

func (c *Cld) GenerateCld() {
	originalImg32FC1 := gocv.NewMatWithSize(c.originalImg.Rows(), c.originalImg.Cols(), gocv.MatTypeCV32F)
	c.originalImg.ConvertTo(&originalImg32FC1, gocv.MatTypeCV32F)

	c.GradientDoG(&originalImg32FC1, &c.dog, Rho, SigmaC)
	c.FlowDoG(&c.dog, &c.fDog, SigmaM)
	c.BinaryThresholding(&c.fDog, &c.result, Tau)
}

func (c *Cld) FlowDoG(src, dst *gocv.Mat, sigma float64) {
	var (
		gauAcc       float64
		gauWeightAcc float64
		gau          []float64
	)

	gausVec := makeGaussianVector(sigma, gau)
	imgWidth, imgHeight := src.Rows(), src.Cols()
	kernelHalf := len(gau) - 1

	for x := 0; x < imgWidth; x++ {
		for y := 0; y < imgHeight; y++ {
			go func(x, y int) {
				c.wg.Add(1)
				gauAcc := -gau[0] * src.GetDoubleAt(x, y)
				gauWeightAcc := -gau[0]

				// Integral alone ETF
				pos := &position{float64(x), float64(y)}
				for step := 0; step < kernelHalf; step++ {
					tmp := c.etf.flowField.GetVecfAt(int(pos.x), int(pos.y))
					direction := &position{x: float64(tmp[0]), y: float64(tmp[1])}

					if direction.x == 0 && direction.y == 0 {
						break
					}

					if pos.x > float64(imgWidth-1) || pos.x < 0.0 ||
						pos.y > float64(imgHeight-1) || pos.y < 0.0 {
						break
					}

					value := src.GetDoubleAt(int(pos.x), int(pos.y))
					weight := gausVec[step]
					gauAcc += value * weight
					gauWeightAcc += weight

					pos.x += direction.x
					pos.y += direction.y

					if int(math.Round(pos.x)) < 0 || int(math.Round(pos.x)) > imgWidth-1 ||
						int(math.Round(pos.y)) < 0 || int(math.Round(pos.y)) > imgHeight-1 {
						break
					}
				}

				// Integral alone inverse ETF
				pos = &position{float64(x), float64(y)}
				for step := 0; step < kernelHalf; step++ {
					tmp := c.etf.flowField.GetVecfAt(int(pos.x), int(pos.y))
					direction := &position{x: float64(-tmp[0]), y: float64(-tmp[1])}

					if direction.x == 0 && direction.y == 0 {
						break
					}

					if pos.x > float64(imgWidth-1) || pos.x < 0.0 ||
						pos.y > float64(imgHeight-1) || pos.y < 0.0 {
						break
					}

					value := src.GetDoubleAt(int(pos.x), int(pos.y))
					weight := gausVec[step]
					gauAcc += value * weight
					gauWeightAcc += weight

					pos.x += direction.x
					pos.y += direction.y

					if int(math.Round(pos.x)) < 0 || int(math.Round(pos.x)) > imgWidth-1 ||
						int(math.Round(pos.y)) < 0 || int(math.Round(pos.y)) > imgHeight-1 {
						break
					}
				}

				c.wg.Done()
			}(x, y)

			newVal := func(gauAcc, gauWeightAcc float64) float64 {
				var res float64

				if gauAcc/gauWeightAcc > 0 {
					res = 1.0
				} else {
					res = 1.0 + math.Tanh(gauAcc/gauWeightAcc)
				}
				return res
			}

			// Update pixel value in the destination matrix.
			dst.SetDoubleAt(x, y, newVal(gauAcc, gauWeightAcc))
		}
	}
	gocv.Normalize(*dst, dst, 0, 1, gocv.NormMixMax)
	c.wg.Wait()
}

func (c *Cld) GradientDoG(src, dst *gocv.Mat, rho, sigmaC float64) {
	var sigmaS = SigmaRatio * sigmaC
	var gauC, gauS []float64
	gvc := makeGaussianVector(sigmaC, gauC)
	gvs := makeGaussianVector(sigmaS, gauS)

	kernel := len(gvs) - 1

	for x := 0; x < dst.Rows(); x++ {
		for y := 0; y < dst.Cols(); y++ {
			go func(x, y int) {
				c.wg.Add(1)

				var (
					gauCAcc, gauSAcc             float64
					gauCWeightAcc, gauSWeightAcc float64
				)

				tmp := c.etf.flowField.GetVecfAt(x, y)
				gradient := position{x: float64(-tmp[0]), y: float64(tmp[1])}

				for step := -kernel; step <= kernel; step++ {
					row := float64(x) + gradient.x*float64(step)
					col := float64(y) + gradient.y*float64(step)

					if row > float64(dst.Rows()-1) || row < 0.0 || col > float64(dst.Cols()-1) || col < 0.0 {
						continue
					}
					val := src.GetFloatAt(int(math.Round(row)), int(math.Round(col)))
					gauIdx := abs(step)

					gauCWeight := func(gauIdx int) float64 {
						if gauIdx >= len(gvc) {
							return 0.0
						}
						return gvc[gauIdx]
					}(gauIdx)

					gauSWeight := gvs[gauIdx]
					gauCAcc += float64(val) * gauCWeight
					gauSAcc += float64(val) * gauSWeight
					gauCWeightAcc += gauCWeight
					gauSWeightAcc += gauSWeight
				}

				vc := gauCAcc / gauCWeightAcc
				vs := gauSAcc / gauSWeightAcc

				res := vc - rho*vs
				dst.SetDoubleAt(x, y, res)

				c.wg.Done()
			}(x, y)
		}
	}
	c.wg.Wait()
}

// BinaryThresholding threshold an image as black and white
func (c *Cld) BinaryThresholding(src, dst *gocv.Mat, tau float64) {
	for x := 0; x < src.Rows(); x++ {
		for y := 0; y < src.Cols(); y++ {
			go func(x, y int) {
				c.wg.Add(1)

				h := src.GetDoubleAt(x, y)
				v := func(h float64) uint8 {
					if h < tau {
						return 0
					}
					return 255
				}(h)
				dst.SetUCharAt(x, y, v)

				c.wg.Done()
			}(x, y)
		}
	}
	c.wg.Wait()
}

func (c *Cld) combineImage() {
	for x := 0; x < c.originalImg.Rows(); x++ {
		for y := 0; y < c.originalImg.Cols(); y++ {
			go func(x, y int) {
				c.wg.Add(1)

				h := c.result.GetUCharAt(x, y)
				if h == 0 {
					c.originalImg.SetUCharAt(x, y, h)
				}
				c.wg.Done()
			}(x, y)
		}
	}

	c.wg.Wait()
}

// gauss computes gaussian function of variance
func gauss(x, mean, sigma float64) float64 {
	return math.Exp((-(x-mean)*(x-mean))/(2*sigma*sigma)) / math.Sqrt(math.Pi*2.0*sigma*sigma)
}

func makeGaussianVector(sigma float64, gau []float64) []float64 {
	var threshold = 0.001
	var i int

	for {
		i++
		if gauss(float64(i), 0.0, sigma) < threshold {
			break
		}
	}
	// clear slice
	gau = gau[:0]
	// extend slice
	gau = append(gau, make([]float64, i+1)...)
	gau[0] = gauss(0.0, 0.0, sigma)

	for j := 1; j < len(gau); j++ {
		gau[j] = gauss(float64(j), 0.0, sigma)
	}

	return gau
}

// abs return the absolute value of x
func abs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}
