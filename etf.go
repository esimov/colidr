package colidr

import (
	"image"
	"log"
	"math"
	"sync"

	"gocv.io/x/gocv"
)

type Etf struct {
	flowField   gocv.Mat
	refinedEtf  gocv.Mat
	gradientMag gocv.Mat
	wg          sync.WaitGroup
	mu          sync.Mutex
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

func (etf *Etf) SetDefaultEtf(file string, size image.Point) error {
	etf.resizeMat(size)

	// Todo check if we should use color or grayscale.
	src := gocv.IMRead(file, gocv.IMReadGrayScale)
	rows, cols := src.Rows(), src.Cols()
	srcNew := gocv.NewMatWithSize(rows, cols, gocv.MatTypeCV32F)
	gocv.Normalize(src, &srcNew, 0.0, 1.0, gocv.NormMinMax)

	gradX := gocv.NewMatWithSize(rows, cols, gocv.MatTypeCV32F)
	gradY := gocv.NewMatWithSize(rows, cols, gocv.MatTypeCV32F)
	gocv.Sobel(srcNew, &gradX, gocv.MatTypeCV32F, 1, 0, 5, 1, 0, gocv.BorderDefault)
	gocv.Sobel(srcNew, &gradY, gocv.MatTypeCV32F, 0, 1, 5, 1, 0, gocv.BorderDefault)

	// Compute gradient
	gocv.Magnitude(gradX, gradY, &etf.gradientMag)
	gocv.Normalize(etf.gradientMag, &etf.gradientMag, 0.0, 1.0, gocv.NormMinMax)

	data := etf.flowField.ToBytes()

	for x := 0; x < cols; x++ {
		for y := 0; y < rows; y++ {
			etf.wg.Add(1)
			go func(x, y int) {
				u := gradX.GetVecfAt(x, y)
				v := gradY.GetVecfAt(x, y)

				// Update the flowField vector with values obtained from sobel matrix.
				idx := (x + (y * rows)) + etf.flowField.Channels()
				data[idx+0] = byte(u[0])
				data[idx+1] = byte(v[0])
				data[idx+2] = 0.0
				data[idx+3] = 0.0

				//flowField.SetTo(gocv.Scalar{Val1: float64(u[0]), Val2: float64(v[0]), Val3: 0.0, Val4: 0.0})
				etf.wg.Done()
			}(x, y)
		}
	}
	nm, err := gocv.NewMatFromBytes(rows, cols, gocv.MatChannels3, data)
	if err != nil {
		log.Fatalf("Cannot create new Mat from bytes: %v", err)
	}
	gocv.Normalize(nm, &etf.flowField, 0.0, 1.0, gocv.NormMinMax)
	etf.rotateFlow(&etf.flowField, 90)

	etf.wg.Wait()
	return nil
}

func (etf *Etf) RefineEtf(kernel int) {
	for x := 0; x < etf.flowField.Rows(); x++ {
		for y := 0; y < etf.flowField.Cols(); y++ {
			// Spawn computation into separate goroutines
			go func(x, y int) {
				etf.wg.Add(1)
				etf.computeNewVector(x, y, kernel)
				etf.wg.Done()
			}(x, y)
		}
	}
	etf.wg.Wait()

	etf.flowField = etf.refinedEtf
}

func (etf *Etf) computeNewVector(x, y int, kernel int) {
	var tNew float32
	tCurX := etf.flowField.GetVecfAt(x, y)

	for r := x - kernel; r <= x+kernel; r++ {
		for c := y - kernel; c <= y+kernel; c++ {
			// Checking for boundaries.
			if r < 0 || r >= etf.refinedEtf.Rows() || c < 0 || c >= etf.refinedEtf.Cols() {
				continue
			}
			tCurY := etf.flowField.GetVecfAt(r, c)
			phi := etf.computePhi(tCurX, tCurY)

			// Compute the euclidean distance of the current point and the neighboring point.
			weightSpatial := etf.computeWeightSpatial(point{x, y}, point{r, c}, kernel)
			weightMagnitude := etf.computeWeightMagnitude(etf.gradientMag.GetFloatAt(x, y), etf.gradientMag.GetFloatAt(r, c))
			weightDirection := etf.computeWeightDirection(tCurX, tCurY)

			tNew += phi * tCurY[0] * weightSpatial * weightMagnitude * weightDirection
		}
	}
	etf.refinedEtf.SetTo(gocv.Scalar{Val1: float64(tNew), Val2: float64(tNew), Val3: float64(tNew), Val4: 0.0})
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

func (etf *Etf) resizeMat(size image.Point) {
	gocv.Resize(etf.flowField, &etf.flowField, size, 0, 0, gocv.InterpolationDefault)
	gocv.Resize(etf.refinedEtf, &etf.refinedEtf, size, 0, 0, gocv.InterpolationDefault)
	gocv.Resize(etf.gradientMag, &etf.gradientMag, size, 0, 0, gocv.InterpolationDefault)
}

func (etf *Etf) rotateFlow(src *gocv.Mat, theta float64) {
	theta = theta / 180.0 * math.Pi
	data := src.ToBytes()

	for x := 0; x < src.Rows(); x++ {
		for y := 0; y < src.Cols(); y++ {
			srcVec := src.GetVecfAt(x, y)

			// Obtain the source vector value and rotate it.
			rx := float64(srcVec[0])*math.Cos(theta) - float64(srcVec[1])*math.Sin(theta)
			ry := float64(srcVec[0])*math.Sin(theta) + float64(srcVec[1])*math.Cos(theta)

			// Apply the rotation values to the destination matrix.
			idx := (x + (y * src.Rows())) + src.Channels()
			data[idx+0] = byte(rx)
			data[idx+1] = byte(ry)
			data[idx+2] = 0.0
			data[idx+3] = 0.0
		}
	}
	nm, err := gocv.NewMatFromBytes(src.Rows(), src.Cols(), gocv.MatChannels3, data)
	if err != nil {
		log.Fatalf("Cannot create new Mat from bytes: %v", err)
	}
	*src = nm.Clone()
}

// normalize normalize two values between 0..1
func normalize(a, b float32) float32 {
	norm := 1 - float32(math.Abs(float64(a)-float64(b))/math.Max(float64(a), float64(b)))
	if norm < 0.0 {
		return 0.0
	}
	return norm
}
