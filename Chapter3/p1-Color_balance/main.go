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
var img2 *ebiten.Image

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

}

type Game struct {
}

func (g *Game) Update() error {
	for row := range 800 {
		for col := range 1422 {
			c := img.At(row, col)
			r, g, b, _ := c.RGBA()
			rr := uint8(r / 256)
			gg := uint8(g / 256)
			bb := uint8(b / 256)
			newc := color.Color(color.RGBA{rr / 2, gg, bb, 0})
			img2.Set(row, col, newc)
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
}

func (g *Game) Layout(outsideWidth, outsideHeight int) (screenWidth, screenHeight int) {
	return 1600, 1422
}

func main() {
	fmt.Println("It's starting!")
	ebiten.SetWindowSize(1600/2, 1422/2)
	ebiten.SetWindowTitle("Problem 1; Color Balance")
	ebiten.SetTPS(0)
	err := ebiten.RunGame(&Game{})
	if err != nil {
		log.Fatal(err)
	}
}
