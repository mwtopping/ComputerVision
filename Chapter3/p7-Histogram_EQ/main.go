package main

import (
	"fmt"
	//"image/color"
	_ "image/png"
	"log"
	"math"

	"github.com/hajimehoshi/ebiten/v2"
	"github.com/hajimehoshi/ebiten/v2/ebitenutil"
)

var img *ebiten.Image
var img2 *ebiten.Image

func init() {
	var err error

	img, _, err = ebitenutil.NewImageFromFile("disney.png")
	if err != nil {
		panic(err)
	}

	img2 = ebiten.NewImage(858, 642)

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
	//	for col := range 910 {
	//		for row := range 732 {
	//			c := img.At(col, row)
	//			r, g, b, _ := c.RGBA()
	//			rr, gg, bb := balancecolor(r, g, b, 1.0, 1.0)
	//			dist := math.Sqrt(math.Pow(float64(rr)-0.0, 2) +
	//				math.Pow(float64(gg)-0.0, 2) +
	//				math.Pow(float64(bb)-255.0, 2))
	//
	//			if (dist < 270.0) && (col < 500) {
	//				newc := color.Color(color.RGBA{255, 255, 255, 255})
	//				img3.Set(row, col, newc)
	//				c2 := img2.At(row, col)
	//				img4.Set(row, col, c2)
	//			}
	//
	//			//			newc := color.Color(color.RGBA{rr, gg, bb, 255})
	//			//			img2.Set(row, col, newc)
	//			//			rr, gg, bb = balancecolor(r, g, b, 1.5, 2.2)
	//
	//		}
	//	}
	fmt.Println("Updated!")
	return nil
}

func (g *Game) Draw(screen *ebiten.Image) {
	screen.DrawImage(img, nil)

	op := &ebiten.DrawImageOptions{}
	op.GeoM.Translate(858, 0)
	screen.DrawImage(img2, op)

}

func (g *Game) Layout(outsideWidth, outsideHeight int) (screenWidth, screenHeight int) {
	return 858 * 2, 642
}

func main() {
	fmt.Println("It's starting!")
	ebiten.SetWindowSize(858*2, 642)
	ebiten.SetWindowTitle("Problem 1; Color Balance")
	ebiten.SetTPS(0)
	err := ebiten.RunGame(&Game{})
	if err != nil {
		log.Fatal(err)
	}
}
