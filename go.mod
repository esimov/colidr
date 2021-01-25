module github.com/esimov/colidr

go 1.12

require (
	gocv.io/x/gocv v0.16.0
	golang.org/x/image v0.0.0-20190703141733-d6a02ce849c9
)

replace gocv.io/x/gocv => ./vendor/gocv.io/x/gocv/
