package main

import (
	"errors"
	"flag"
	"fmt"
	"image/jpeg"
	"image/png"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	"github.com/esimov/colidr"
	"gocv.io/x/gocv"
	"golang.org/x/image/bmp"
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
		sigmaR        = flag.Float64("sr", 2.6, "SigmaR")
		sigmaM        = flag.Float64("sm", 3.0, "SigmaM")
		sigmaC        = flag.Float64("sc", 1.0, "SigmaC")
		rho           = flag.Float64("rho", 0.98, "Rho")
		tau           = flag.Float64("tau", 0.98, "Tau")
		etfKernel     = flag.Int("k", 3, "Etf kernel")
		etfIteration  = flag.Int("ei", 1, "Number of Etf iteration")
		fDogIteration = flag.Int("di", 0, "Number of FDoG iteration")
		blurSize      = flag.Int("bl", 3, "Blur size")
		antiAlias     = flag.Bool("aa", false, "Anti aliasing")
		visEtf        = flag.Bool("ve", false, "Visualize Etf")
		visResult     = flag.Bool("vr", false, "Visualize end result")
		potrace       = flag.Bool("pt", true, "Use potrace to smooth edges")
	)

	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, fmt.Sprintf(banner, Version))
		flag.PrintDefaults()
	}
	flag.Parse()

	if len(*source) == 0 || len(*destination) == 0 {
		log.Fatal("Usage: colidr -in <source> -out <destination>")
	}

	fileTypes := []string{".jpg", ".jpeg", ".png", ".bmp"}
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
		EtfKernel:     *etfKernel,
		EtfIteration:  *etfIteration,
		FDogIteration: *fDogIteration,
		BlurSize:      *blurSize,
		AntiAlias:     *antiAlias,
		VisEtf:        *visEtf,
		VisResult:     *visResult,
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

	if *potrace {
		*destination = strings.Replace(*destination, ext, ".bmp", 1)
		ext = filepath.Ext(*destination)
	}

	// save the image byte array to the destination file
	output, err := os.Create(*destination)
	if err != nil {
		log.Fatalf("error saving the image: %v", err)
	}

	switch ext {
	case ".jpg", ".jpeg":
		err = jpeg.Encode(output, img, &jpeg.Options{Quality: 100})
	case ".png":
		err = png.Encode(output, img)
	case ".bmp":
		done := make(chan bool)
		go func(err error) {
			err = bmp.Encode(output, img)
			if err == nil {
				done <- true
			}
		}(err)

		if *potrace {
			for {
				select {
				case <-done:
					dir := filepath.Dir(*destination)
					file := filepath.Base(output.Name()) + ".pgm"
					dest := dir + "/" + file
					args := []string{"-g", *destination, "-o", dest}
					cmd := exec.Command("potrace", args...)

					if _, err = cmd.Output(); err != nil {
						fmt.Fprintln(os.Stderr, "potrace error: ", err)
						os.Exit(1)
					}
					fmt.Println("finished")
					return
				}
			}
		}
	default:
		err = errors.New("unsupported image format")
	}
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
