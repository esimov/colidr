package main

import (
	"flag"
	"fmt"
	"image/png"
	"log"
	"os"
	"path/filepath"
	"time"

	"github.com/esimov/colidr"
	"gocv.io/x/gocv"
)

const banner = `
┌─┐┌─┐┬  ┬┌┬┐┬─┐
│  │ ││  │ ││├┬┘
└─┘└─┘┴─┘┴─┴┘┴└─

Coherent Line Drawing CLI
    Version: %s

`

// Version indicates the current build version.
var Version string

func main() {
	var (
		source        = flag.String("in", "", "Source image")
		destination   = flag.String("out", "", "Destination image")
		sigmaR        = flag.Float64("r", 2.6, "SigmaR")
		sigmaM        = flag.Float64("m", 3.0, "SigmaM")
		sigmaC        = flag.Float64("c", 1.0, "SigmaC")
		rho           = flag.Float64("rho", 0.98, "Rho")
		tau           = flag.Float64("tau", 0.98, "Tau")
		etfKernel = flag.Int("kernel", 1, "Etf kernel")
		etfIteration = flag.Int("eit", 2, "Number of Etf iteration")
		fDogIteration = flag.Int("fit", 0, "Number of FDoG iteration")
		blurSize      = flag.Int("blur", 3, "Blur size")
		antiAlias     = flag.Bool("aa", false, "Anti aliasing")
		etfViz        = flag.Bool("etf", false, "Vizualize Etf")
		flowField     = flag.Bool("ff", false, "Vizualize flowfield")
	)

	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, fmt.Sprintf(banner, Version))
		flag.PrintDefaults()
	}
	flag.Parse()

	if len(*source) == 0 || len(*destination) == 0 {
		log.Fatal("Usage: colidr -in <source> -out <destination>")
	}

	fileTypes := []string{".jpg", ".jpeg", ".png"}
	ext := filepath.Ext(*destination)

	if !supportedFiles(ext, fileTypes) {
		log.Fatalf("Output file type not supported: %v", ext)
	}

	opts := colidr.Options{
		SigmaR:        *sigmaR,
		SigmaM:        *sigmaM,
		SigmaC:        *sigmaC,
		Rho:           *rho,
		Tau:           float32(*tau),
		EtfKernel: *etfKernel,
		EtfIteration: *etfIteration,
		FDogIteration: *fDogIteration,
		BlurSize:      *blurSize,
		AntiAlias:     *antiAlias,
		EtfViz:        *etfViz,
		FlowField:     *flowField,
	}

	fmt.Print("Generating")

	start := time.Now()
	done := make(chan struct{})

	ticker := time.NewTicker(time.Millisecond * 100)
	go func() {
		for {
			select {
			case <-ticker.C:
				fmt.Print(".")
			case <-done:
				ticker.Stop()
				end := time.Now().Sub(start)
				fmt.Printf("\nDone in: %.2fs\n", end.Seconds())
			}
		}
	}()

	cld, err := colidr.NewCLD(*source, opts)
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
	out, err := os.Create(*destination)
	if err != nil {
		log.Fatalf("error saving the image: %v", err)
	}

	err = png.Encode(out, img)
	done <- struct{}{}

	time.Sleep(time.Second)
}

// supportedFiles checks if the provided file extension is supported.
func supportedFiles(ext string, types []string) bool {
	for _, t := range types {
		if t == ext {
			return true
		}
	}
	return false
}
