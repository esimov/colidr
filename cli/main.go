package main

import (
	"log"

	"github.com/esimov/colidr"
)

func main() {
	opts := colidr.Options{
		SigmaR: 1.6,
		SigmaM: 3.0,
		SigmaC: 1.0,
		Rho:    0.997,
		Tau:    0.8,
	}

	_, err := colidr.NewCLD("input.jpg", opts)
	if err != nil {
		log.Fatalf("Cannot initialize Coherent Line Drawing: %v", err)
	}
}
