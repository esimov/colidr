package colidr

import (
	"errors"
	"fmt"
	"image"
	"math"
	"os"
	"sync"

	"gocv.io/x/gocv"
)

type Cld struct {
	srcImage gocv.Mat
	result   gocv.Mat
	dog      gocv.Mat
	fDog     gocv.Mat
	etf      *Etf
	wg       sync.WaitGroup
	Options
}

type Options struct {
	SigmaR float64
	SigmaM float64
	SigmaC float64
	Rho    float64
	Tau    float64
}

type position struct {
	x, y float64
}

func NewCLD(imgFile string, cldOpts Options) (*Cld, error) {
	if _, err := os.Stat(imgFile); os.IsNotExist(err) {
		return nil, err
	}

	srcImage := gocv.IMRead(imgFile, gocv.IMReadGrayScale)
	cols, rows := srcImage.Cols(), srcImage.Rows()

	result := gocv.NewMatWithSize(rows, cols, gocv.MatTypeCV8UC1)
	dog := gocv.NewMatWithSize(rows, cols, gocv.MatTypeCV32F)
	fDog := gocv.NewMatWithSize(rows, cols, gocv.MatTypeCV32F)

	var wg sync.WaitGroup

	etf := NewETF()
	etf.Init(rows, cols)

	err := etf.InitDefaultEtf(imgFile, image.Point{X: rows, Y: cols})
	if err != nil {
		return nil, errors.New(fmt.Sprintf("Unable to initialize edge tangent flow: %s\n", err))
	}
	return &Cld{
		srcImage, result, dog, fDog, etf, wg, cldOpts,
	}, nil
}

func (c *Cld) GenerateCld() (int, int, []byte) {
	srcImg32FC1 := gocv.NewMatWithSize(c.srcImage.Rows(), c.srcImage.Cols(), gocv.MatTypeCV32F)
	c.srcImage.ConvertTo(&srcImg32FC1, gocv.MatTypeCV32F, 1.0/255.0)

	c.GradientDoG(&srcImg32FC1, &c.dog, c.Rho, c.SigmaC)
	c.FlowDoG(&c.dog, &c.fDog, c.SigmaM)

	return c.srcImage.Rows(), c.srcImage.Cols(), c.BinaryThreshold(&c.fDog, &c.result, c.Tau)
}

func (c *Cld) GradientDoG(src, dst *gocv.Mat, rho, sigmaC float64) {
	var sigmaS = c.SigmaR * sigmaC
	gvc := makeGaussianVector(sigmaC)
	gvs := makeGaussianVector(sigmaS)
	kernel := len(gvs) - 1

	fmt.Println(dst.Rows(), dst.Cols())
	fmt.Println(c.etf.flowField.Rows(), c.etf.flowField.Cols())

	width, height := dst.Cols(), dst.Rows()
	c.wg.Add(width * (height-1))

	for y := 0; y < height-1; y++ {
		for x := 0; x < width; x++ {
			go func(y, x int) {
				var (
					gauCAcc, gauSAcc             float64
					gauCWeightAcc, gauSWeightAcc float64
				)

				tmp := c.etf.flowField.GetVecfAt(y, x)
				gradient := position{x: float64(-tmp[0]), y: float64(tmp[1])}

				for step := -kernel; step <= kernel; step++ {
					row := float64(y) + gradient.y*float64(step)
					col := float64(x) + gradient.x*float64(step)

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
				dst.SetDoubleAt(y, x, res)

				c.wg.Done()
			}(y, x)
		}
	}
	c.wg.Wait()
	//fmt.Println(dst.ToBytes())
}

func (c *Cld) FlowDoG(src, dst *gocv.Mat, sigma float64) {
	var (
		gauAcc       float64
		gauWeightAcc float64
	)

	gausVec := makeGaussianVector(sigma)
	width, height := src.Cols(), src.Rows()
	kernelHalf := len(gausVec) - 1

	fmt.Println(src.Rows(), src.Cols())
	c.wg.Add(width * (height-1))

	for y := 0; y < height-1; y++ {
		for x := 0; x < width; x++ {
			go func(y, x int) {
				gauAcc = -gausVec[0] * src.GetDoubleAt(y, x)
				gauWeightAcc = -gausVec[0]

				// Integral alone ETF
				pos := &position{float64(x), float64(y)}
				for step := 0; step < kernelHalf; step++ {
					tmp := c.etf.flowField.GetVecfAt(int(pos.y), int(pos.x))
					direction := &position{x: float64(tmp[1]), y: float64(tmp[0])}

					if direction.x == 0 && direction.y == 0 {
						break
					}

					if pos.x > float64(width-1) || pos.x < 0.0 ||
						pos.y > float64(height-1) || pos.y < 0.0 {
						break
					}

					value := src.GetDoubleAt(int(pos.y), int(pos.x))
					weight := gausVec[step]

					gauAcc += value * weight
					gauWeightAcc += weight

					pos.x += direction.x
					pos.y += direction.y

					if int(math.Round(pos.x)) < 0 || int(math.Round(pos.x)) > width-1 ||
						int(math.Round(pos.y)) < 0 || int(math.Round(pos.y)) > height-1 {
						break
					}
				}

				// Integral alone inverse ETF
				pos = &position{float64(x), float64(y)}
				for step := 0; step < kernelHalf; step++ {
					tmp := c.etf.flowField.GetVecfAt(int(pos.y), int(pos.x))
					direction := &position{x: float64(-tmp[1]), y: float64(-tmp[0])}

					if direction.x == 0 && direction.y == 0 {
						break
					}

					if pos.x > float64(width-1) || pos.x < 0.0 ||
						pos.y > float64(height-1) || pos.y < 0.0 {
						break
					}

					value := src.GetDoubleAt(int(pos.y), int(pos.x))
					weight := gausVec[step]

					gauAcc += value * weight
					gauWeightAcc += weight

					pos.x += direction.x
					pos.y += direction.y

					if int(math.Round(pos.x)) < 0 || int(math.Round(pos.x)) > width-1 ||
						int(math.Round(pos.y)) < 0 || int(math.Round(pos.y)) > height-1 {
						break
					}
				}

				c.wg.Done()
			}(y, x)

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
			dst.SetFloatAt(y, x, float32(newVal(gauAcc, gauWeightAcc)))
		}
	}
	fmt.Println("FINAL:", dst.Rows(), dst.Cols())
	gocv.Normalize(*dst, dst, 0.0, 1.0, gocv.NormMinMax)
	//fmt.Println(dst.ToBytes())

	c.wg.Wait()
}

// BinaryThreshold threshold an image as black and white.
func (c *Cld) BinaryThreshold(src, dst *gocv.Mat, tau float64) []byte {
	width, height := dst.Cols(), dst.Rows()
	c.wg.Add(width * (height-1))

	fmt.Println(src.Rows(), src.Cols())
	fmt.Println(dst.Rows(), dst.Cols())
	fmt.Println(width, height)
	for y := 0; y < height-1; y++ {
		for x := 0; x < width; x++ {
			go func(y, x int) {
				h := src.GetDoubleAt(y, x)
				v := func(h float64) uint8 {
					if h < tau {
						return 0
					}
					return 255
				}(h)
				dst.SetUCharAt(y, x, v)

				c.wg.Done()
			}(y, x)
		}
	}
	fmt.Println("====================================================================")
	//fmt.Println(dst.ToBytes())
	c.wg.Wait()

	return dst.ToBytes()
}

func (c *Cld) CombineImage() {
	for x := 0; x < c.srcImage.Rows(); x++ {
		for y := 0; y < c.srcImage.Cols(); y++ {
			c.wg.Add(1)
			go func(x, y int) {
				h := c.result.GetUCharAt(x, y)
				if h == 0 {
					c.srcImage.SetUCharAt(x, y, h)
				}
				c.wg.Done()
			}(x, y)
		}
	}

	// Blur the image a little bit
	gocv.GaussianBlur(c.srcImage, &c.srcImage, image.Point{3, 3}, 0.0, 0.0, gocv.BorderDefault)
	c.wg.Wait()
}

// gauss computes gaussian function of variance
func gauss(x, mean, sigma float64) float64 {
	return math.Exp((-(x-mean)*(x-mean))/(2*sigma*sigma)) / math.Sqrt(math.Pi*2.0*sigma*sigma)
}

func makeGaussianVector(sigma float64) []float64 {
	var (
		gau       []float64
		threshold = 0.001
		i         int
	)

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
