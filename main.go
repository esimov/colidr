package main

import (
	"os"

	"gocv.io/x/gocv"
)

func main() {
	// parse args
	img := gocv.IMRead(os.Args[1], gocv.IMReadGrayScale)
	dst := gocv.NewMatWithSize(img.Rows(), img.Cols(), gocv.MatTypeCV32F)
	gocv.Sobel(img, &dst, gocv.MatTypeCV32F, 0, 1, 5, 1, 0, gocv.BorderDefault)
	gocv.Sobel(img, &dst, gocv.MatTypeCV32F, 1, 0, 5, 1, 0, gocv.BorderDefault)
	gocv.IMWrite("output2.jpg", dst)
}