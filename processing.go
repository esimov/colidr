package colidr

import (
	"image"
	"image/color"
	"math"
	"sync"

	"gocv.io/x/gocv"
)

type PostProcessing struct {
	Etf
	blurSize int
}

func NewPostProcessing(blurSize int) *PostProcessing {
	return &PostProcessing{
		blurSize: blurSize,
	}
}

func (pp *PostProcessing) VizEtf(flowField, dst *gocv.Mat) {
	noise := gocv.NewMatWithSize(flowField.Cols()/2, flowField.Rows()/2, gocv.MatTypeCV32F)

	gocv.Randu(&noise, 0, 1.0)
	gocv.Resize(noise, &noise, image.Point{noise.Rows(), noise.Cols()}, 0, 0, gocv.InterpolationNearestNeighbor)

	s := 5
	rows := noise.Cols()
	cols := noise.Rows()
	sigma := 2 * s * s

	var wg sync.WaitGroup
	wg.Add(rows * cols)

	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			go func(i, j int) {
				defer wg.Done()
				wSum := 0.0
				x := i
				y := j

				for k := 0; k < s; k++ {
					v := flowField.GetVecfAt((x+rows)%rows, (y+cols)%cols)
					if v[0] != 0 {
						v0a := math.Abs(float64(v[0]))
						v1a := math.Abs(float64(v[1]))
						xt := float64(x) + (v0a/v0a+v1a)*(v0a/v0a)
						x = int(xt)
					}
					if v[1] != 0 {
						v0a := math.Abs(float64(v[0]))
						v1a := math.Abs(float64(v[1]))
						yt := float64(y) + (v1a/v0a+v1a)*(v1a/v1a)
						y = int(yt)
					}
					r2 := float64(k * k)
					w := (1.0 / (math.Pi * float64(sigma))) * math.Exp(-(r2)/float64(sigma))

					xx := (x + rows) % rows
					yy := (y + cols) % cols

					dstAt := dst.GetFloatAt(i, j)
					noiseAt := noise.GetFloatAt(xx, yy)
					newVal := float64(dstAt) + (w * float64(noiseAt))

					dst.SetFloatAt(i, j, float32(newVal))

					wSum += w
				}
				x = i
				y = j

				for k := 0; k < s; k++ {
					v := flowField.GetVecfAt((x+rows)%rows, (y+cols)%cols)

					if -v[0] != 0 {
						v0a := math.Abs(float64(-v[0]))
						v1a := math.Abs(float64(-v[1]))
						xt := float64(x) + (v0a/v0a+v1a)*(v0a/v0a)
						x = int(xt)
					}
					if -v[1] != 0 {
						v0a := math.Abs(float64(-v[0]))
						v1a := math.Abs(float64(-v[1]))
						yt := float64(y) + (v1a/v0a+v1a)*(v1a/v1a)
						y = int(yt)
					}
					r2 := float64(k * k)
					w := (1.0 / (math.Pi * float64(sigma))) * math.Exp(-(r2)/float64(sigma))

					xx := (x + rows) % rows
					yy := (y + cols) % cols

					dstAt := dst.GetFloatAt(i, j)
					noiseAt := noise.GetFloatAt(xx, yy)
					newVal := float64(dstAt) + (w * float64(noiseAt))
					dst.SetFloatAt(i, j, float32(newVal))

					wSum += w
				}
				dstAt := dst.GetFloatAt(i, j)
				dstAt /= float32(wSum)

				dst.SetFloatAt(i, j, dstAt)
			}(i, j)
		}
	}

	wg.Wait()
}

func (pp *PostProcessing) FlowField(flowField, dst *gocv.Mat) {
	var resolution = 10

	for i := 0; i < dst.Rows(); i += resolution {
		for j := 0; j < dst.Cols(); j += resolution {
			v := flowField.GetVecfAt(i, j)
			p1 := &point{x: j, y: j}
			p2 := &point{x: i + int(v[0])*2, y: j + int(v[1])*2}

			gocv.ArrowedLine(
				dst,
				image.Point{X: p1.x, Y: p1.y}, image.Point{X: p2.x, Y: p2.y},
				color.RGBA{255, 0, 0, 255},
				1,
			)
		}
	}
}

func (pp *PostProcessing) AntiAlias(src, dst gocv.Mat) {
	gocv.Normalize(src, &dst, 0.0, 255.0, gocv.NormMinMax)
	gocv.GaussianBlur(dst, &dst, image.Point{pp.blurSize, pp.blurSize}, 0.0, 0.0, gocv.BorderConstant)
}
