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
}

func NewPP() *PostProcessing {
	return &PostProcessing{}
}

func (pp *PostProcessing) VisualizeEtf(dis gocv.Mat) *gocv.Mat {
	noise := gocv.NewMatWithSize(pp.flowField.Rows()/2, pp.flowField.Cols()/2, gocv.MatTypeCV32F)
	dst := gocv.NewMatWithSize(pp.flowField.Rows(), pp.flowField.Cols(), gocv.MatTypeCV32F)

	dis.CopyTo(&dst)
	gocv.Randu(&noise, 0, 1.0)
	gocv.Resize(noise, &noise, image.Point{noise.Rows(), noise.Cols()}, 0, 0, gocv.InterpolationNearestNeighbor)

	s := 10
	nRows := noise.Rows()
	nCols := noise.Cols()
	sigma := 2 * s * s

	var wg sync.WaitGroup

	for i := 0; i < nRows; i++ {
		for j := 0; j < nCols; j++ {
			wg.Add(1)
			go func(i, j int) {
				defer wg.Done()
				wSum := 0.0
				x := i
				y := j

				for k := 0; k < s; k++ {
					ffv := pp.flowField.GetFloatAt((x+nRows)%nRows, (y+nCols)%nCols)
					ffm := gocv.NewMatWithSizeFromScalar(
						gocv.Scalar{Val1: float64(ffv), Val2: float64(ffv), Val3: float64(ffv), Val4: float64(ffv)},
						pp.flowField.Rows(),
						pp.flowField.Cols(),
						gocv.MatTypeCV32F,
					)
					gocv.Normalize(ffm, &ffm, 0.0, 1.0, gocv.NormMinMax)

					v := ffm.GetVecfAt((x+nRows)%nRows, (y+nCols)%nCols)
					if v[0] != 0 {
						x = x + (abs(int(v[0]))/abs(int(v[0]))+abs(int(v[1])))*(abs(int(v[0]))/abs(int(v[0])))
					}
					if v[1] != 0 {
						y = y + (abs(int(v[1]))/abs(int(v[0]))+abs(int(v[1])))*(abs(int(v[1]))/abs(int(v[1])))
					}
					r2 := float64(k * k)
					w := (1.0 / (math.Pi * float64(sigma))) * math.Exp(-(r2)/float64(sigma))
					xx := (int(x) + nRows) % nRows
					yy := (int(y) + nCols) % nCols

					dstAt := dst.GetFloatAt(i, j)
					noiseAt := noise.GetFloatAt(xx, yy)
					newVal := float64(dstAt) + (w * float64(noiseAt))
					dst.SetFloatAt(i, j, float32(newVal))

					wSum += w
				}
				x = i
				y = j

				for k := 0; k < s; k++ {
					ffv := pp.flowField.GetFloatAt((x+nRows)%nRows, (y+nCols)%nCols)
					ffm := gocv.NewMatFromScalar(gocv.Scalar{Val1: float64(ffv), Val2: float64(ffv), Val3: float64(ffv), Val4: float64(ffv)}, gocv.MatTypeCV32F)
					gocv.Normalize(ffm, &ffm, 0.0, 1.0, gocv.NormMinMax)

					v := ffm.GetVecfAt((x+nRows)%nRows, (y+nCols)%nCols)
					if v[0] != 0 {
						x = x + (abs(int(v[0]))/abs(int(v[0]))+abs(int(v[1])))*(abs(int(v[0]))/abs(int(v[0])))
					}
					if v[1] != 0 {
						y = y + (abs(int(v[1]))/abs(int(v[0]))+abs(int(v[1])))*(abs(int(v[1]))/abs(int(v[1])))
					}
					r2 := float64(k * k)
					w := (1.0 / (math.Pi * float64(sigma))) * math.Exp(-(r2)/float64(sigma))

					dstAt := dst.GetFloatAt(i, j)
					noiseAt := noise.GetFloatAt((x+nRows)%nRows, (y+nCols)%nCols)
					newVal := float64(dstAt) + (w * float64(noiseAt))
					dst.SetFloatAt(i, j, float32(newVal))

					wSum += w
				}
				dstAt := dst.GetFloatAt(i, j)
				dstAt /= float32(wSum)

				dis = gocv.NewMatWithSizeFromScalar(
					gocv.Scalar{Val1: float64(dstAt), Val2: float64(dstAt), Val3: float64(dstAt), Val4: float64(dstAt)},
					pp.flowField.Rows(),
					pp.flowField.Cols(),
					gocv.MatTypeCV32F,
				)
			}(i, j)
		}
	}
	wg.Wait()

	return &dst
}

// Todo maybe should return Mat.
func (pp *PostProcessing) FlowField(dis *gocv.Mat) {
	var resolution = 10

	for i := 0; i < pp.flowField.Rows(); i += resolution {
		for j := 0; j < pp.flowField.Cols(); j += resolution {
			v := pp.flowField.GetVecfAt(i, j)
			p1 := &point{x: int(v[i]), y: int(v[j])}
			p2 := &point{x: i + int(v[0])*5, y: j + int(v[1])*5}

			gocv.ArrowedLine(
				dis,
				image.Point{X: p1.x, Y: p1.y}, image.Point{X: p2.x, Y: p2.y},
				color.RGBA{255, 0, 0, 255},
				1,
			)
		}
	}
}

func (pp *PostProcessing) AntiAlias(src gocv.Mat, dst gocv.Mat) {
	var blurSize = 3

	gocv.Normalize(src, &dst, 60.0, 255.0, gocv.NormMinMax)
	gocv.GaussianBlur(dst, &dst, image.Point{blurSize, blurSize}, 0.0, 0.0, gocv.BorderDefault)
}
