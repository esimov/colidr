package main

import (
	"image/png"
	"log"
	"os"

	"github.com/esimov/colidr"
	"gocv.io/x/gocv"
)

func main() {
	opts := colidr.Options{
		SigmaR:    1.6,
		SigmaM:    4.55,
		SigmaC:    1.612,
		Rho:       1.994,
		Tau:       0.58,
		BlurSize:  3,
		AntiAlias: true,
	}

	cld, err := colidr.NewCLD("lena.jpg", opts)
	if err != nil {
		log.Fatalf("cannot initialize CLD: %v", err)
	}

	data := cld.GenerateCld()

	rows, cols := cld.Image.Rows(), cld.Image.Cols()
	mat, err := gocv.NewMatFromBytes(rows, cols, gocv.MatTypeCV8UC1, data)
	if err != nil {
		log.Fatalf("error retrieving the byte array: %v", err)
	}

	img, err := mat.ToImage()
	if err != nil {
		log.Fatalf("error converting matrix to image: %v", err)
	}

	//save the imgByte to file
	out, err := os.Create("output.png")
	if err != nil {
		log.Fatalf("error saving the image: %v", err)
	}

	err = png.Encode(out, img)
}
