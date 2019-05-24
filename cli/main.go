package main

import (
	"log"

	"github.com/esimov/colidr"
	"os"
	"image/png"
	"gocv.io/x/gocv"
)

func main() {
	opts := colidr.Options{
		SigmaR: 1.6,
		SigmaM: 4.55,
		SigmaC: 1.8112,
		Rho:    1.997,
		Tau:    0.000781,
	}

	cld, err := colidr.NewCLD("me.jpeg", opts)
	if err != nil {
		log.Fatalf("cannot initialize CLD: %v", err)
	}

	width, height, data := cld.GenerateCld()

	mat, err := gocv.NewMatFromBytes(width, height, gocv.MatTypeCV8UC1, data)
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
