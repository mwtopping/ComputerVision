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
var img2 *ebiten.Image
var img3 *ebiten.Image

func init() {
	var err error
	img, _, err = ebitenutil.NewImageFromFile("flowers.jpg")
	if err != nil {
		log.Fatal(err)
	}
	img2, _, err = ebitenutil.NewImageFromFile("flowers.jpg")
	if err != nil {
		log.Fatal(err)
	}
	img3, _, err = ebitenutil.NewImageFromFile("flowers.jpg")
	if err != nil {
		log.Fatal(err)
	}

}

func clamp(x, vmin, vmax uint32) (y uint32) {
	var clamped uint32
	if x < vmin {
		clamped = vmin
	} else if x > vmax {
		clamped = vmax
	} else {
		clamped = x
	}

	return clamped
}

func balancecolor(r, g, b uint32, scale, power float64) (nr, ng, nb uint8) {

	// assumes the input jpg is input as with uint32 values for RGB
	_r := math.Pow(math.Pow(float64(r/256), power)*scale, 1/power)
	_g := math.Pow(math.Pow(float64(g/256), power)*scale, 1/power)
	_b := math.Pow(math.Pow(float64(b/256), power)*scale, 1/power)

	rr := uint8(clamp(uint32(_r), 0, 255))
	gg := uint8(clamp(uint32(_g), 0, 255))
	bb := uint8(clamp(uint32(_b), 0, 255))

	return rr, gg, bb
}

type Game struct {
}

func (g *Game) Update() error {
	for row := range 800 {
		for col := range 1422 {
			c := img.At(row, col)
			r, g, b, _ := c.RGBA()
			rr, gg, bb := balancecolor(r, g, b, 1.5, 1.0)
			newc := color.Color(color.RGBA{rr, gg, bb, 255})
			img2.Set(row, col, newc)
			rr, gg, bb = balancecolor(r, g, b, 1.5, 2.2)
			newc = color.Color(color.RGBA{rr, gg, bb, 255})
			img3.Set(row, col, newc)

		}
	}
	fmt.Println("Updated!")
	return nil
}

func (g *Game) Draw(screen *ebiten.Image) {
	screen.DrawImage(img, nil)

	op := &ebiten.DrawImageOptions{}
	op.GeoM.Translate(800, 0)
	screen.DrawImage(img2, op)

	op2 := &ebiten.DrawImageOptions{}
	op2.GeoM.Translate(1600, 0)
	screen.DrawImage(img3, op2)

}

func (g *Game) Layout(outsideWidth, outsideHeight int) (screenWidth, screenHeight int) {
	return 2400, 1422
}

func main() {
	fmt.Println("It's starting!")
	ebiten.SetWindowSize(2400/2, 1422/2)
	ebiten.SetWindowTitle("Problem 1; Color Balance")
	ebiten.SetTPS(0)
	err := ebiten.RunGame(&Game{})
	if err != nil {
		log.Fatal(err)
	}
}
