package main

import (
	"image"

	"gocv.io/x/gocv"
	"math"
	"sync"
	"github.com/go-opencv/go-opencv/opencv"
)

type Etf struct {
	flowField   gocv.Mat
	newEtf      gocv.Mat
	gradientMag gocv.Mat
}

func NewETF() *Etf {
	return &Etf{}
}

func (etf *Etf) Init(row, col int) {
	size := gocv.NewMatWithSize(row, col, gocv.MatTypeCV32F)

	etf.flowField = gocv.NewMatWithSize(size.Rows(), size.Cols(), gocv.MatTypeCV32F)
	etf.newEtf = gocv.NewMatWithSize(size.Rows(), size.Cols(), gocv.MatTypeCV32F)
	etf.gradientMag = gocv.NewMatWithSize(size.Rows(), size.Cols(), gocv.MatTypeCV32F)
}

func (etf *Etf) intializeEtf(file string, size gocv.Mat) error {
	etf.resizeMat(size)

	src := gocv.IMRead(file, gocv.IMReadUnchanged)
	dst := gocv.NewMat()
	gocv.Normalize(src, &dst, 0.0, 1.0, gocv.NormMixMax)

	newImg, err := dst.ToImage()
	if err != nil {
		return err
	}

	// ToDo apply different x, y derivatives
	// ToDo convert to grayscale
	sobelGradX := Sobel(newImg.(*image.NRGBA), 10)
	sobelGradY := Sobel(newImg.(*image.NRGBA), 10)

	gradX, err := gocv.ImageToMatRGBA(sobelGradX)
	if err != nil {
		return err
	}
	gradY, err := gocv.ImageToMatRGBA(sobelGradY)
	if err != nil {
		return err
	}

	// Compute gradient
	gocv.Magnitude(gradX, gradY, &etf.gradientMag)
	gocv.Normalize(etf.gradientMag, &etf.gradientMag, 0.0, 1.0, gocv.NormMixMax)

	flowField := gocv.NewMat()


	var wg sync.WaitGroup

	for x := 0; x < src.Rows(); x++ {
		for y := 0; y < src.Cols(); y++ {
			go func(x, y int) {
				wg.Add(1)

				u := gradX.GetVecfAt(x, y)
				v := gradY.GetVecfAt(x, y)

				flowField.SetFloatAt(x, y, normalize(u[0], v[0]))

				wg.Done()
			}(x, y)
		}
	}
	wg.Wait()

	return nil
}

func (etf *Etf) resizeMat(size gocv.Mat) {
	gocv.Resize(etf.flowField, &etf.flowField, image.Point{size.Rows(), size.Cols()}, 0, 0, gocv.InterpolationDefault)
	gocv.Resize(etf.newEtf, &etf.newEtf, image.Point{size.Rows(), size.Cols()}, 0, 0, gocv.InterpolationDefault)
	gocv.Resize(etf.gradientMag, &etf.gradientMag, image.Point{size.Rows(), size.Cols()}, 0, 0, gocv.InterpolationDefault)
}

func (etf *Etf) rotateFlow(src, dst gocv.Mat, theta float32) {
	theta = theta / 180.0 * math.Pi

	for x := 0; x < src.Rows(); x++ {
		for y := 0; y < src.Cols(); y++ {
			val := src.GetVecfAt(x, y)
			rx := val[0] * math.Cos(theta) - val[1] * math.Sin(theta)
			ry := val[0] * math.Sin(theta) + val[1] * math.Cos(theta)
			dst.SetFloatAt(x, y, gocv.Vecf{rx, ry})
		}
	}
}

func main() {
	etf := NewETF()
	etf.Init(300, 300)
}

func normalize(a, b float32) float32 {
	norm := 1 - float32(math.Abs(float64(a)-float64(b)) / math.Max(float64(a), float64(b)))
	if norm < 0.0 {
		return 0.0
	}
	return norm
}