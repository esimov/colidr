package colidr

import (
	"image"
	"math"
	"sync"

	"gocv.io/x/gocv"
)

type Etf struct {
	flowField   gocv.Mat
	refinedEtf  gocv.Mat
	gradientMag gocv.Mat
	wg          sync.WaitGroup
	mu          sync.RWMutex
}

type point struct {
	x int
	y int
}

func NewETF() *Etf {
	return &Etf{}
}

func (etf *Etf) Init(rows, cols int) {
	etf.flowField = gocv.NewMatWithSize(rows, cols, gocv.MatChannels3)
	etf.refinedEtf = gocv.NewMatWithSize(rows, cols, gocv.MatChannels3)
	etf.gradientMag = gocv.NewMatWithSize(rows, cols, gocv.MatChannels3)
}

func (etf *Etf) InitDefaultEtf(file string, size image.Point) error {
	etf.resizeMat(size)

	src := gocv.IMRead(file, gocv.IMReadColor)
	gocv.Normalize(src, &src, 0.0, 1.0, gocv.NormMinMax)

	gradX := gocv.NewMatWithSize(src.Rows(), src.Cols(), gocv.MatTypeCV32F)
	gradY := gocv.NewMatWithSize(src.Rows(), src.Cols(), gocv.MatTypeCV32F)

	gocv.Sobel(src, &gradX, gocv.MatTypeCV32F, 1, 0, 5, 1, 0, gocv.BorderDefault)
	gocv.Sobel(src, &gradY, gocv.MatTypeCV32F, 0, 1, 5, 1, 0, gocv.BorderDefault)

	// Compute gradient
	gocv.Magnitude(gradX, gradY, &etf.gradientMag)
	gocv.Normalize(etf.gradientMag, &etf.gradientMag, 0.0, 1.0, gocv.NormMinMax)

	width, height := src.Cols(), src.Rows()
	etf.wg.Add(width * height)

	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			go func(y, x int) {
				etf.mu.RLock()
				defer etf.mu.RUnlock()

				u := gradX.GetVecfAt(y, x)
				v := gradY.GetVecfAt(y, x)

				etf.flowField.SetVecfAt(y, x, gocv.Vecf{u[0], v[0], 0.0})
				etf.wg.Done()
			}(y, x)
		}
	}

	etf.wg.Wait()

	gocv.Normalize(etf.flowField, &etf.flowField, 0.0, 1.0, gocv.NormMinMax)
	etf.rotateFlow(&etf.flowField, &etf.flowField, 90)

	return nil
}

func (etf *Etf) RefineEtf(kernel int) {
	for y := 0; y < etf.flowField.Rows(); y++ {
		for x := 0; x < etf.flowField.Cols(); x++ {
			etf.wg.Add(1)
			// Spawn computation into separate goroutines
			go func(x, y int) {
				etf.mu.Lock()
				defer etf.mu.Unlock()

				etf.computeNewVector(x, y, kernel)
				etf.wg.Done()
			}(x, y)
		}
	}
	etf.wg.Wait()
	etf.flowField = etf.refinedEtf
}

func (etf *Etf) resizeMat(size image.Point) {
	gocv.Resize(etf.flowField, &etf.flowField, size, 0, 0, gocv.InterpolationDefault)
	gocv.Resize(etf.refinedEtf, &etf.refinedEtf, size, 0, 0, gocv.InterpolationDefault)
	gocv.Resize(etf.gradientMag, &etf.gradientMag, size, 0, 0, gocv.InterpolationDefault)
}

func (etf *Etf) computeNewVector(x, y int, kernel int) {
	var tNew float32
	tCurX := etf.flowField.GetVecfAt(y, x)

	for r := y - kernel; r <= y+kernel; r++ {
		for c := x - kernel; c <= x+kernel; c++ {
			// Checking for boundaries.
			if r < 0 || r >= etf.refinedEtf.Rows() || c < 0 || c >= etf.refinedEtf.Cols() {
				continue
			}
			tCurY := etf.flowField.GetVecfAt(r, c)
			phi := etf.computePhi(tCurX, tCurY)

			// Compute the euclidean distance of the current point and the neighboring point.
			weightSpatial := etf.computeWeightSpatial(point{x, y}, point{c, r}, kernel)
			weightMagnitude := etf.computeWeightMagnitude(etf.gradientMag.GetFloatAt(y, x), etf.gradientMag.GetFloatAt(r, c))
			weightDirection := etf.computeWeightDirection(tCurX, tCurY)

			tNew += phi * tCurY[0] * weightSpatial * weightMagnitude * weightDirection
		}
	}
	etf.refinedEtf.SetFloatAt(y, x, tNew)
	gocv.Normalize(etf.refinedEtf, &etf.refinedEtf, 0.0, 1.0, gocv.NormMinMax)
}

func (etf *Etf) computePhi(x, y gocv.Vecf) float32 {
	wd := etf.computeWeightDirection(x, y)
	if wd > 0 {
		return 1.0
	}
	return -1.0
}

func (etf *Etf) computeWeightSpatial(p1, p2 point, r int) float32 {
	// Get the euclidean distance of two points.
	dx := p2.x - p1.x
	dy := p2.y - p1.y

	dist := math.Sqrt(float64(dx*dx) + float64(dy*dy))
	if dist < float64(r) {
		return 1.0
	}
	return 0.0
}

func (etf *Etf) computeWeightMagnitude(gradMagX, gradMagY float32) float32 {
	return (1.0 + float32(math.Tanh(float64(gradMagX-gradMagY)))) / 2.0
}

func (etf *Etf) computeWeightDirection(x, y gocv.Vecf) float32 {
	var s float32
	// Compute the dot product.
	for i := 0; i < etf.flowField.Channels(); i++ {
		s += x[i] * y[i]
	}
	return float32(math.Abs(float64(s)))
}

func (etf *Etf) rotateFlow(src, dst *gocv.Mat, theta float64) {
	theta = theta / 180.0 * math.Pi

	width, height := src.Cols(), src.Rows()
	etf.wg.Add(width * height)

	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			go func(y, x int) {
				etf.mu.RLock()
				defer etf.mu.RUnlock()

				srcVec := src.GetVecfAt(y, x)

				// Obtain the source vector value and rotate it.
				rx := float64(srcVec[0])*math.Cos(theta) - float64(srcVec[1])*math.Sin(theta)
				ry := float64(srcVec[0])*math.Sin(theta) + float64(srcVec[1])*math.Cos(theta)

				dst.SetVecfAt(y, x, gocv.Vecf{float32(rx), float32(ry), 0.0})
				etf.wg.Done()
			}(y, x)
		}
	}
	etf.wg.Wait()
}

// normalize normalize two values between 0..1
func normalize(a, b float32) float32 {
	norm := 1 - float32(math.Abs(float64(a)-float64(b))/math.Max(float64(a), float64(b)))
	if norm < 0.0 {
		return 0.0
	}
	return norm
}
