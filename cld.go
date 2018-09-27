package main

import (
	"errors"
	"fmt"
	"gocv.io/x/gocv"
	"math"
	"sync"
)

type Cld struct {
	originalImg gocv.Mat
	result      gocv.Mat
	dog         gocv.Mat
	fDog        gocv.Mat
	etf         *Etf
}

type position struct {
	x, y float64
}

const (
	SigmaM float64 = 3.0
	SigmaC float64 = 1.0
	Rho    float64 = 0.997
	Tau    float64 = 0.8
)

func NewCLD() *Cld {
	return &Cld{}
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

}

func (c *Cld) FlowDoG(src, dst gocv.Mat, sigma float64) {
	var (
		gauAcc       float64
		gauWeightAcc float64
		gau          []float64
	)

	gausVec := makeGaussianVector(sigma, gau)
	imgWidth, imgHeight := src.Rows(), src.Cols()
	kernelHalf := len(gau) - 1

	var wg sync.WaitGroup

	for x := 0; x < imgWidth; x++ {
		for y := 0; y < imgHeight; y++ {
			go func(x, y int) {
				wg.Add(1)
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

					if int(pos.x) < 0 || int(pos.x) > imgWidth-1 || int(pos.y) < 0 || int(pos.y) > imgHeight-1 {
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

					if int(pos.x) < 0 || int(pos.x) > imgWidth-1 || int(pos.y) < 0 || int(pos.y) > imgHeight-1 {
						break
					}
				}

				wg.Done()
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
	gocv.Normalize(dst, &dst, 0, 1, gocv.NormMixMax)
	wg.Wait()
}

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
