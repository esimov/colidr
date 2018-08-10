package main

import (
	"gocv.io/x/gocv"
	"image"
)

type Etf struct {
	flowField gocv.Mat
	newEtf gocv.Mat
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

func (etf *Etf) intializeEtf(file string, size gocv.Mat) {
	etf.resize(size)

	src := gocv.IMRead(file, gocv.IMReadUnchanged)
	dst := gocv.NewMat()
	gocv.Normalize(src, &dst, 0.0, 1.0, gocv.NormMixMax)

	//ToDo implement Sobel image transformation
}


func (etf *Etf) resize(size gocv.Mat) {
	gocv.Resize(etf.flowField, &etf.flowField, image.Point{size.Rows(), size.Rows()}, 0, 0, gocv.InterpolationDefault)
	gocv.Resize(etf.newEtf, &etf.newEtf, image.Point{size.Rows(), size.Rows()}, 0, 0, gocv.InterpolationDefault)
	gocv.Resize(etf.gradientMag, &etf.gradientMag, image.Point{size.Rows(), size.Rows()}, 0, 0, gocv.InterpolationDefault)
}


func main() {
	etf := NewETF()
	etf.Init(300, 300)
}
