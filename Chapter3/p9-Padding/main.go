package main

import (
	"fmt"
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
	padval = 16
}

func pad(origImg *ebiten.Image, pad int) (paddedImage *ebiten.Image) {
	fmt.Println(pad)
	p := origImg.Bounds().Size()
	w := p.X
	h := p.Y
	fmt.Println(origImg.Bounds().Size())
	fmt.Println(w, h)

	newimg := ebiten.NewImage(800+2*pad, 1422+2*pad)

	for col := range w {
		for row := range h {
			c := img.At(col, row)
			//			r, g, b, _ := c.RGBA()
			//			rr, gg, bb := balancecolor(r, g, b, 1.5, 1.0)
			//			newc := color.Color(color.RGBA{rr, gg, bb, 255})
			newimg.Set(col+pad, row+pad, c)
		}
	}

	return newimg
}

//func clamp(x, vmin, vmax uint32) (y uint32) {
//	var clamped uint32
//	if x < vmin {
//		clamped = vmin
//	} else if x > vmax {
//		clamped = vmax
//	} else {
//		clamped = x
//	}
//
//	return clamped
//}
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
}

func (g *Game) Update() error {

	g.paddedImg = pad(img, padval)

	//	for row := range 800 {
	//		for col := range 1422 {
	//			c := img.At(row, col)
	//			r, g, b, _ := c.RGBA()
	//			rr, gg, bb := balancecolor(r, g, b, 1.5, 1.0)
	//			newc := color.Color(color.RGBA{rr, gg, bb, 255})
	//
	//		}
	//	}
	fmt.Println("Updated!")
	return nil
}

func (g *Game) Draw(screen *ebiten.Image) {

	op := &ebiten.DrawImageOptions{}
	op.GeoM.Scale(800/float64(800+2*padval), 1422/float64(1422+2*padval))
	screen.DrawImage(g.paddedImg, op)

}

func (g *Game) Layout(outsideWidth, outsideHeight int) (screenWidth, screenHeight int) {
	//	p := g.paddedImg.Bounds().Size()
	//	w := p.X
	//	h := p.Y

	return 800, 1422
}

func main() {
	fmt.Println("It's starting!")
	ebiten.SetWindowSize(800/2, 1422/2)
	ebiten.SetWindowTitle("Problem 1; Color Balance")
	ebiten.SetTPS(0)
	err := ebiten.RunGame(&Game{})
	if err != nil {
		log.Fatal(err)
	}
}
