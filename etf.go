package main

import (
	"image"

	"gocv.io/x/gocv"
	"math"
	"sync"
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

func (etf *Etf) rotateFlow(src, dst gocv.Mat, theta float64) {
	theta = theta / 180.0 * math.Pi

	for x := 0; x < src.Rows(); x++ {
		for y := 0; y < src.Cols(); y++ {
			srcVec := src.GetVecfAt(x, y)
			// Obtain the source vector value and rotate it.
			rx := srcVec[0] * float32(math.Cos(theta)) - srcVec[1] * float32(math.Sin(theta))
			ry := srcVec[0] * float32(math.Sin(theta)) + srcVec[1] * float32(math.Cos(theta))

			// Apply the rotation values to the destination matrix.
			dstVec := dst.GetVecfAt(x, y)
			dstVec[0], dstVec[1] = rx, ry
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