package main

import (
	"fmt"
	"gocv.io/x/gocv"
)

type PostProcessing struct {
	flowField gocv.Mat
	dis gocv.Mat
}

func NewPP(rows, cols int) *PostProcessing {
	noise := gocv.NewMatWithSize(rows/2, cols/2, gocv.MatTypeCV32F)
	//dis := gocv.NewMatWithSize(pp.flowField.Rows(), pp.flowField.Cols(), gocv.MatTypeCV32F)
	gocv.Randu(&noise, 0, 1.0)
	fmt.Println(noise.GetFloatAt(10, 10))

	return &PostProcessing{}
}