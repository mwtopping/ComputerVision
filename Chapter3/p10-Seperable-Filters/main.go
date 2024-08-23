package main

import (
	"fmt"
	"image/color"
	_ "image/jpeg"
	"log"

	"github.com/hajimehoshi/ebiten/v2"
	"github.com/hajimehoshi/ebiten/v2/ebitenutil"
)

var img *ebiten.Image
var padval int

func init() {
	var err error
	img, _, err = ebitenutil.NewImageFromFile("flowers.jpg")
	if err != nil {
		log.Fatal(err)
	}
	padval = 1
}

func pad(origImg *ebiten.Image, pad int) (paddedImage *ebiten.Image) {
	p := origImg.Bounds().Size()
	w := p.X
	h := p.Y

	newimg := ebiten.NewImage(800+2*pad, 1422+2*pad)

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
	paddedImg *ebiten.Image
	finalImg  *ebiten.Image
}

func convolve(inpImg *ebiten.Image, kernelsize int) (outputimg *ebiten.Image) {
	p := inpImg.Bounds().Size()
	w := p.X
	h := p.Y

	fmt.Println("Starting Horizontal Convolution")
	padsize := kernelsize/2 - 1
	himg := ebiten.NewImage(w-2*padsize, h-2*padsize)

	hkernel := make([]int32, kernelsize)
	hkernel[0] = -1
	hkernel[1] = 0
	hkernel[2] = 1

	for row := range w {
		for col := range h {

			finalr := int32(0)
			finalb := int32(0)
			finalg := int32(0)
			//			fmt.Println("For Pixel ", row, col)
			for ii := range kernelsize {
				//				fmt.Println(row+kernelsize+ii-kernelsize/2, col+kernelsize)
				c := inpImg.At(row+padsize+ii-kernelsize/2, col+padsize)
				r, g, b, _ := c.RGBA()
				finalr += hkernel[ii] * int32(r/256)
				finalg += hkernel[ii] * int32(g/256)
				finalb += hkernel[ii] * int32(b/256)

			}

			finalr /= int32(2)
			finalg /= int32(2)
			finalb /= int32(2)

			newc := color.Color(color.RGBA{uint8(clamp(finalr, 0, 255)),
				uint8(clamp(finalg, 0, 255)),
				uint8(clamp(finalb, 0, 255)),
				255})
			himg.Set(row, col, newc)
		}
	}

	seconditer := pad(himg, padsize)

	fmt.Println("Starting Vertical Convolution")

	newnewimg := ebiten.NewImage(w-2*padsize, h-2*padsize)

	vkernel := make([]int32, kernelsize)
	vkernel[0] = 1
	vkernel[1] = 2
	vkernel[2] = 1

	for row := range w {
		for col := range h {

			finalr := int32(0)
			finalb := int32(0)
			finalg := int32(0)
			//			fmt.Println("For Pixel ", row, col)
			for ii := range kernelsize {
				c := seconditer.At(row+padsize, col+padsize+ii-kernelsize/2)
				r, g, b, _ := c.RGBA()
				finalr += vkernel[ii] * int32(r/256)
				finalg += vkernel[ii] * int32(g/256)
				finalb += vkernel[ii] * int32(b/256)

			}

			finalr /= int32(4)
			finalg /= int32(4)
			finalb /= int32(4)

			newc := color.Color(color.RGBA{uint8(clamp(finalr+127, 0, 255)),
				uint8(clamp(finalg+127, 0, 255)),
				uint8(clamp(finalb+127, 0, 255)),
				255})
			newnewimg.Set(row, col, newc)
		}
	}

	return newnewimg
}

func (g *Game) Update() error {

	g.paddedImg = pad(img, padval)
	g.finalImg = convolve(g.paddedImg, 2*padval+1)

	fmt.Println("Updated!")
	return nil
}

func (g *Game) Draw(screen *ebiten.Image) {

	screen.DrawImage(img, nil)

	op := &ebiten.DrawImageOptions{}
	op.GeoM.Translate(800, 0)
	screen.DrawImage(g.finalImg, op)

}

func (g *Game) Layout(outsideWidth, outsideHeight int) (screenWidth, screenHeight int) {

	return 1600, 1422
}

func main() {
	fmt.Println("It's starting!")
	ebiten.SetWindowSize(1600/2, 1422/2)
	ebiten.SetWindowTitle("Problem 10; Separable Filters")
	ebiten.SetTPS(0)
	err := ebiten.RunGame(&Game{})
	if err != nil {
		log.Fatal(err)
	}
}
