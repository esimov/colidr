package colidr

import (
	"image"
	"math"
	"sync"

	"gocv.io/x/gocv"
	"fmt"
)

type Etf struct {
	flowField   gocv.Mat
	gradientField gocv.Mat
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
	etf.flowField = gocv.NewMatWithSize(rows, cols, gocv.MatTypeCV32F + gocv.MatChannels3)
	etf.gradientField = gocv.NewMatWithSize(rows, cols, gocv.MatTypeCV32F + gocv.MatChannels3)
	etf.refinedEtf = gocv.NewMatWithSize(rows, cols, gocv.MatTypeCV32F + gocv.MatChannels3)
	etf.gradientMag = gocv.NewMatWithSize(rows, cols, gocv.MatTypeCV32F + gocv.MatChannels3)
}

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

	/*window := gocv.NewWindow("gradx")
	window.IMShow(gradX)
	window.WaitKey(0)

	window = gocv.NewWindow("grady")
	window.IMShow(gradY)
	window.WaitKey(0)*/

	// Compute gradient
	gocv.Magnitude(gradX, gradY, &etf.gradientMag)
	gocv.Normalize(etf.gradientMag, &etf.gradientMag, 0.0, 1.0, gocv.NormMinMax)

	//gradX.ConvertTo(&gradX, gocv.MatTypeCV8UC3, 255)
	//gradY.ConvertTo(&gradY, gocv.MatTypeCV8UC3, 255)

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
				idx := y*ch + (x*ch*height)

				data[idx+0] = byte(v[0])
				data[idx+1] = byte(u[0])
				data[idx+2] = 0.0

				//fmt.Println(gocv.Vecb{v[0], u[0], 0.0})
				etf.gradientField.SetVecfAt(y, x, gocv.Vecf{v[0], u[0], 0})
				//fmt.Println(gocv.Vecf{v[0], u[0], 0})
				etf.wg.Done()
			}(y, x)
		}
	}

	etf.wg.Wait()

	window := gocv.NewWindow("gradient")
	window.IMShow(etf.gradientField)
	window.WaitKey(0)

	//etf.gradientField.ConvertTo(&etf.gradientField, gocv.MatTypeCV32F + gocv.MatChannels3, 255)
	fmt.Println(etf.gradientField.Type())
	etf.rotateFlow(&etf.gradientField, &etf.flowField, 90)
	//etf.flowField.ConvertTo(&etf.flowField, gocv.MatTypeCV64F, 255)

	window = gocv.NewWindow("flow")
	window.IMShow(etf.flowField)
	window.WaitKey(0)

	gocv.IMWrite("/home/esimov/Desktop/flowfield.tiff", etf.flowField)

	return nil
}

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

func (etf *Etf) resizeMat(size image.Point) {
	gocv.Resize(etf.gradientField, &etf.gradientField, size, 0, 0, gocv.InterpolationLinear)
	gocv.Resize(etf.flowField, &etf.flowField, size, 0, 0, gocv.InterpolationLinear)
	gocv.Resize(etf.refinedEtf, &etf.refinedEtf, size, 0, 0, gocv.InterpolationLinear)
	gocv.Resize(etf.gradientMag, &etf.gradientMag, size, 0, 0, gocv.InterpolationLinear)
}

func (etf *Etf) computeNewVector(x, y int, kernel int) {
	var tNew0, tNew1, tNew2 float32
	tCurX := etf.flowField.GetVecfAt(y, x)
	//fmt.Println(tCurX)

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

func (etf *Etf) computePhi(x, y gocv.Vecf) float32 {
	dot := etf.computeDot(x, y)
	if dot > 0 {
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
	return float32(math.Abs(float64(etf.computeDot(x, y))))
}

func (etf *Etf) computeDot(x, y gocv.Vecf) float32 {
	var s float32
	ch := etf.flowField.Channels()

	for i := 0; i < ch; i++ {
		s += x[i] * y[i]
	}
	return s
}

func (etf *Etf) normalize(x, y, z float32) gocv.Vecf {
	nv := float32(math.Sqrt(float64(x*x) + float64(y*y) + float64(z*z)))

	if nv > 0.0 {
		return gocv.Vecf{x * 1.0/nv, y * 1.0/nv, z * 1.0/nv}
	}
	return gocv.Vecf{0.0, 0.0, 0.0}
}

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

// normalize normalize two values between 0..1
func normalize(a, b float32) float32 {
	norm := 1 - float32(math.Abs(float64(a)-float64(b))/math.Max(float64(a), float64(b)))
	if norm < 0.0 {
		return 0.0
	}
	return norm
}
