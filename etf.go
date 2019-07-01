package colidr

import (
	"image"
	"math"
	"sync"

	"gocv.io/x/gocv"
)

// Etf is the main entry struct for the edge tangent flow computation.
// It encompass the basic operational entities needed for the matrix operations.
type Etf struct {
	flowField     gocv.Mat
	gradientField gocv.Mat
	refinedEtf    gocv.Mat
	gradientMag   gocv.Mat
	wg            sync.WaitGroup
	mu            sync.RWMutex
}

// point is a basic struct for vector type operations
type point struct {
	x int
	y int
}

// NewETF is a constructor method which initializes an Etf struct.
func NewETF() *Etf {
	return &Etf{}
}

// Init initializes the ETF matrices.
func (etf *Etf) Init(rows, cols int) {
	etf.flowField = gocv.NewMatWithSize(rows, cols, gocv.MatTypeCV32F+gocv.MatChannels3)
	etf.gradientField = gocv.NewMatWithSize(rows, cols, gocv.MatTypeCV32F+gocv.MatChannels3)
	etf.refinedEtf = gocv.NewMatWithSize(rows, cols, gocv.MatTypeCV32F+gocv.MatChannels3)
	etf.gradientMag = gocv.NewMatWithSize(rows, cols, gocv.MatTypeCV32F+gocv.MatChannels3)
}

// InitDefaultEtf computes the gradientField matrix by setting up
// the pixel values from original image on which a sobel threshold has been applied.
func (etf *Etf) InitDefaultEtf(file string, size image.Point) error {
	etf.resizeMat(size)

	src := gocv.IMRead(file, gocv.IMReadColor)
	src.ConvertTo(&src, gocv.MatTypeCV32F, 255)
	gocv.Normalize(src, &src, 0.0, 1.0, gocv.NormMinMax)

	// Generate gradX and gradY
	gradX := gocv.NewMatWithSize(src.Rows(), src.Cols(), gocv.MatTypeCV32F)
	gradY := gocv.NewMatWithSize(src.Rows(), src.Cols(), gocv.MatTypeCV32F)

	gocv.Sobel(src, &gradX, gocv.MatTypeCV32F, 1, 0, 5, 1, 0, gocv.BorderDefault)
	gocv.Sobel(src, &gradY, gocv.MatTypeCV32F, 0, 1, 5, 1, 0, gocv.BorderDefault)

	// Compute gradient
	gocv.Magnitude(gradX, gradY, &etf.gradientMag)
	gocv.Normalize(etf.gradientMag, &etf.gradientMag, 0.0, 1.0, gocv.NormMinMax)

	data := etf.gradientField.ToBytes()
	ch := etf.gradientField.Channels()

	width, height := src.Cols(), src.Rows()
	etf.wg.Add(width * height)

	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			go func(y, x int) {
				etf.mu.RLock()
				defer etf.mu.RUnlock()

				u := gradX.GetVecfAt(y, x)
				v := gradY.GetVecfAt(y, x)

				// Obtain the pixel channel value from Mat image and
				// update the gradientField vector with values from sobel matrix.
				idx := y*ch + (x * ch * height)

				data[idx+0] = byte(v[0])
				data[idx+1] = byte(u[0])
				data[idx+2] = 0.0

				etf.gradientField.SetVecfAt(y, x, gocv.Vecf{v[0], u[0], 0})
				etf.wg.Done()
			}(y, x)
		}
	}

	etf.wg.Wait()

	window := gocv.NewWindow("gradient")
	window.IMShow(etf.gradientField)
	window.WaitKey(0)

	etf.rotateFlow(&etf.gradientField, &etf.flowField, 90)

	window = gocv.NewWindow("flow")
	window.IMShow(etf.flowField)
	window.WaitKey(0)

	return nil
}

// RefineEtf will compute the refined edge tangent flow
// based on the formulas from the original paper.
func (etf *Etf) RefineEtf(kernel int) {
	width, height := etf.flowField.Cols(), etf.flowField.Rows()
	etf.wg.Add(width * height)

	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			// Spawn computation into separate goroutines
			go func(y, x int) {
				etf.mu.Lock()
				etf.computeNewVector(y, x, kernel)
				etf.mu.Unlock()

				etf.wg.Done()
			}(y, x)
		}
	}
	etf.wg.Wait()
	etf.flowField = etf.refinedEtf.Clone()
}

// resizeMat resize all the matrices
func (etf *Etf) resizeMat(size image.Point) {
	gocv.Resize(etf.gradientField, &etf.gradientField, size, 0, 0, gocv.InterpolationLinear)
	gocv.Resize(etf.flowField, &etf.flowField, size, 0, 0, gocv.InterpolationLinear)
	gocv.Resize(etf.refinedEtf, &etf.refinedEtf, size, 0, 0, gocv.InterpolationLinear)
	gocv.Resize(etf.gradientMag, &etf.gradientMag, size, 0, 0, gocv.InterpolationLinear)
}

// rotateFlow applies a rotation on the original gradient field and calculates the new angles.
func (etf *Etf) rotateFlow(src, dst *gocv.Mat, theta float64) {
	theta = theta / 180.0 * math.Pi

	width, height := src.Cols(), src.Rows()
	etf.wg.Add(width * height)

	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			go func(y, x int) {
				etf.mu.Lock()
				defer etf.mu.Unlock()

				v := src.GetVecfAt(y, x)

				// Obtain the vector value and rotate it.
				rx := float64(v[0])*math.Cos(theta) - float64(v[1])*math.Sin(theta)
				ry := float64(v[0])*math.Sin(theta) + float64(v[1])*math.Cos(theta)

				dst.SetVecfAt(y, x, gocv.Vecf{float32(rx), float32(ry), 0})

				etf.wg.Done()
			}(y, x)
		}
	}
	etf.wg.Wait()
}

// computeNewVector computes a new, normalized vector from the refined edge tangent flow matrix following the original paper Eq(1).
func (etf *Etf) computeNewVector(x, y int, kernel int) {
	var tNew0, tNew1, tNew2 float32
	tCurX := etf.flowField.GetVecfAt(y, x)

	for r := y - kernel; r <= y+kernel; r++ {
		for c := x - kernel; c <= x+kernel; c++ {
			// Checking for boundaries.
			if r < 0 || r >= etf.refinedEtf.Rows() || c < 0 || c >= etf.refinedEtf.Cols() {
				continue
			}
			tCurY := etf.flowField.GetVecfAt(r, c)

			phi := etf.computePhi(tCurX, tCurY)
			// Compute the euclidean distance of the current point and the neighborhood point.
			ws := etf.computeWeightSpatial(point{x, y}, point{c, r}, kernel)
			wm := etf.computeWeightMagnitude(etf.gradientMag.GetFloatAt(y, x), etf.gradientMag.GetFloatAt(r, c))
			wd := etf.computeWeightDirection(tCurX, tCurY)

			tNew0 += phi * tCurY[0] * ws * wm * wd
			tNew1 += phi * tCurY[1] * ws * wm * wd
			tNew2 += phi * tCurY[2] * ws * wm * wd
		}
	}

	etf.refinedEtf.SetVecfAt(y, x, etf.normalize(tNew0, tNew1, tNew2))
}

// computeWeightSpatial implementation of Paper's Eq(2)
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

// computeWeightMagnitude implementation of Paper's Eq(3)
func (etf *Etf) computeWeightMagnitude(gradMagX, gradMagY float32) float32 {
	return (1.0 + float32(math.Tanh(float64(gradMagX-gradMagY)))) / 2.0
}

// computeWeightDirection implementation of Paper's Eq(4)
func (etf *Etf) computeWeightDirection(x, y gocv.Vecf) float32 {
	return float32(math.Abs(float64(etf.computeDot(x, y))))
}

// computePhi implementation of Paper's Eq(5)
func (etf *Etf) computePhi(x, y gocv.Vecf) float32 {
	dot := etf.computeDot(x, y)
	if dot > 0 {
		return 1.0
	}
	return -1.0
}

// computeDot computes the dot product of two vectors
func (etf *Etf) computeDot(x, y gocv.Vecf) float32 {
	var s float32
	ch := etf.flowField.Channels()

	for i := 0; i < ch; i++ {
		s += x[i] * y[i]
	}
	return s
}

// normalize returns a normalized vector
func (etf *Etf) normalize(x, y, z float32) gocv.Vecf {
	nv := float32(math.Sqrt(float64(x*x) + float64(y*y) + float64(z*z)))

	if nv > 0.0 {
		return gocv.Vecf{x * 1.0 / nv, y * 1.0 / nv, z * 1.0 / nv}
	}
	return gocv.Vecf{0.0, 0.0, 0.0}
}
