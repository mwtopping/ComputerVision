package main

import (
	"fmt"
	"image/color"
	_ "image/jpeg"
	"log"
	"math"

	"github.com/hajimehoshi/ebiten/v2"
	"github.com/hajimehoshi/ebiten/v2/ebitenutil"
)

var img *ebiten.Image
var padval int

func init() {
	var err error
	img, _, err = ebitenutil.NewImageFromFile("th.jpg")
	if err != nil {
		log.Fatal(err)
	}
	padval = 2
}

func pad(origImg *ebiten.Image, pad int) (paddedImage *ebiten.Image) {
	p := origImg.Bounds().Size()
	w := p.X
	h := p.Y

	newimg := ebiten.NewImage(320+2*pad, 320+2*pad)

	for col := range w {
		for row := range h {
			c := origImg.At(col, row)
			//			r, g, b, _ := c.RGBA()
			//			rr, gg, bb := balancecolor(r, g, b, 1.5, 1.0)
			//			newc := color.Color(color.RGBA{rr, gg, bb, 255})
			newimg.Set(col+pad, row+pad, c)
		}
	}

	return newimg
}

func clamp(x, vmin, vmax int32) (y int32) {
	var clamped int32
	if x < vmin {
		clamped = vmin
	} else if x > vmax {
		clamped = vmax
	} else {
		clamped = x
	}

	return clamped
}

//
//func balancecolor(r, g, b uint32, scale, power float64) (nr, ng, nb uint8) {
//
//	// assumes the input jpg is input as with uint32 values for RGB
//	_r := math.Pow(math.Pow(float64(r/256), power)*scale, 1/power)
//	_g := math.Pow(math.Pow(float64(g/256), power)*scale, 1/power)
//	_b := math.Pow(math.Pow(float64(b/256), power)*scale, 1/power)
//
//	rr := uint8(clamp(uint32(_r), 0, 255))
//	gg := uint8(clamp(uint32(_g), 0, 255))
//	bb := uint8(clamp(uint32(_b), 0, 255))
//
//	return rr, gg, bb
//}

type Game struct {
	finalImg *ebiten.Image
	gaussImg *ebiten.Image
}

func gaussianConvolve(inpImg *ebiten.Image, kernelsize int) (outputimg *ebiten.Image) {

	paddedImg := pad(inpImg, (kernelsize-1)/2)

	p := paddedImg.Bounds().Size()
	w := p.X
	h := p.Y

	padsize := (kernelsize - 1) / 2
	himg := ebiten.NewImage(w-2*padsize, h-2*padsize)

	kernel := [5][5]int32{
		{1, 4, 6, 4, 1},
		{4, 16, 24, 16, 4},
		{6, 24, 36, 24, 6},
		{4, 16, 24, 16, 4},
		{1, 4, 6, 4, 1},
	}

	for row := range w {
		for col := range h {

			finalr := int32(0)
			finalb := int32(0)
			finalg := int32(0)
			//			fmt.Println("For Pixel ", row, col)
			for ii := range kernelsize {
				for jj := range kernelsize {
					c := paddedImg.At(row+padsize+ii-kernelsize/2, col+padsize+jj-kernelsize/2)
					r, g, b, _ := c.RGBA()
					finalr += kernel[ii][jj] * int32(r/256)
					finalg += kernel[ii][jj] * int32(g/256)
					finalb += kernel[ii][jj] * int32(b/256)
				}

			}

			finalr /= int32(256)
			finalg /= int32(256)
			finalb /= int32(256)

			newc := color.Color(color.RGBA{uint8(clamp(finalr, 0, 255)),
				uint8(clamp(finalr, 0, 255)),
				uint8(clamp(finalr, 0, 255)),
				255})
			himg.Set(row, col, newc)
		}
	}

	return himg
}

func kernelsum(kernel [5][5]float64, kernelsize int) (sum float64) {
	total := float64(0)
	for ii := range kernelsize {
		for jj := range kernelsize {
			total += kernel[ii][jj]
		}
	}
	return total
}

func bilateralFilter(inpImg *ebiten.Image, kernelsize int) (outputimg *ebiten.Image) {

	paddedImg := pad(inpImg, (kernelsize-1)/2)

	p := paddedImg.Bounds().Size()
	w := p.X
	h := p.Y

	fmt.Println("Starting Horizontal Convolution")
	padsize := (kernelsize - 1) / 2
	himg := ebiten.NewImage(w-2*padsize, h-2*padsize)

	var kernel [5][5]float64
	sigmad := 2.5
	sigmar := 40.0

	for row := range w {
		for col := range h {
			c := paddedImg.At(row, col)
			ithisr, _, _, _ := c.RGBA()
			thisr := float64(ithisr / 256)

			finalr := float64(0)
			finalb := float64(0)
			finalg := float64(0)
			//			fmt.Println("For Pixel ", row, col)
			for ii := range kernelsize {
				for jj := range kernelsize {
					c := paddedImg.At(row+padsize+ii-kernelsize/2, col+padsize+jj-kernelsize/2)
					ithatr, _, _, _ := c.RGBA()
					thatr := float64(ithatr / 256)

					domainval := -0.5 * (math.Pow(float64(ii), 2) +
						math.Pow(float64(jj), 2)) / math.Pow(sigmad, 2)
					rangeval := -0.5 * (math.Pow(thisr-thatr, 2)) / math.Pow(sigmar, 2)

					exponent := domainval + rangeval

					kernelval := math.Exp(exponent)
					kernel[ii][jj] = kernelval
					//					fmt.Println(thisr, thatr, rangeval)
				}
			}
			ksum := kernelsum(kernel, 5)

			for ii := range kernelsize {
				for jj := range kernelsize {
					c := paddedImg.At(row+padsize+ii-kernelsize/2, col+padsize+jj-kernelsize/2)
					r, g, b, _ := c.RGBA()
					finalr += kernel[ii][jj] * float64(r/256)
					finalg += kernel[ii][jj] * float64(g/256)
					finalb += kernel[ii][jj] * float64(b/256)
				}

			}
			newc := color.Color(color.RGBA{uint8(clamp(int32(finalr/ksum), 0, 255)),
				uint8(clamp(int32(finalr/ksum), 0, 255)),
				uint8(clamp(int32(finalr/ksum), 0, 255)),
				255})
			himg.Set(row, col, newc)
		}
	}
	return himg
}

func (g *Game) Update() error {
	g.finalImg = bilateralFilter(img, 2*padval+1)
	g.gaussImg = gaussianConvolve(img, 2*padval+1)

	fmt.Println("Updated!")
	return nil
}

func (g *Game) Draw(screen *ebiten.Image) {

	screen.DrawImage(img, nil)

	op := &ebiten.DrawImageOptions{}
	op.GeoM.Translate(320, 0)
	screen.DrawImage(g.gaussImg, op)

	op = &ebiten.DrawImageOptions{}
	op.GeoM.Translate(640, 0)
	screen.DrawImage(g.finalImg, op)

}

func (g *Game) Layout(outsideWidth, outsideHeight int) (screenWidth, screenHeight int) {

	return 320 * 3, 320
}

func main() {
	fmt.Println("It's starting!")
	ebiten.SetWindowSize(320*4.5, 320*1.5)
	ebiten.SetWindowTitle("Problem 12; Bilateral Filtering")
	ebiten.SetTPS(0)
	err := ebiten.RunGame(&Game{})
	if err != nil {
		log.Fatal(err)
	}
}
