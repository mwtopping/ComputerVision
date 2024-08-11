package main

import (
	"fmt"
	//	"image/color"
	_ "image/jpeg"
	"log"
	//	"math"

	"github.com/hajimehoshi/ebiten/v2"
	"github.com/hajimehoshi/ebiten/v2/ebitenutil"
)

var img *ebiten.Image
var img2 *ebiten.Image
var img3 *ebiten.Image

func init() {
	var err error
	img, _, err = ebitenutil.NewImageFromFile("FG.jpg")
	if err != nil {
		log.Fatal(err)
	}
	img2, _, err = ebitenutil.NewImageFromFile("BG.jpg")
	if err != nil {
		log.Fatal(err)
	}

	img3, _, err = ebitenutil.NewImageFromFile("FG.jpg")
	if err != nil {
		log.Fatal(err)
	}

}

type Game struct {
}

func (g *Game) Update() error {
	//	for row := range 910 {
	//		for col := range 732 {
	//			c := img.At(row, col)
	//			r, g, b, _ := c.RGBA()
	//			//			rr, gg, bb := balancecolor(r, g, b, 1.5, 1.0)
	//			//			newc := color.Color(color.RGBA{rr, gg, bb, 255})
	//			//			img2.Set(row, col, newc)
	//			//			rr, gg, bb = balancecolor(r, g, b, 1.5, 2.2)
	//			//			newc = color.Color(color.RGBA{rr, gg, bb, 255})
	//			//			img3.Set(row, col, newc)
	//
	//		}
	//	}
	fmt.Println("Updated!")
	return nil
}

func (g *Game) Draw(screen *ebiten.Image) {
	screen.DrawImage(img, nil)

	op := &ebiten.DrawImageOptions{}
	op.GeoM.Translate(910, 0)
	screen.DrawImage(img2, op)

	op2 := &ebiten.DrawImageOptions{}
	op2.GeoM.Translate(910*2, 0)
	screen.DrawImage(img3, op2)

}

func (g *Game) Layout(outsideWidth, outsideHeight int) (screenWidth, screenHeight int) {
	return 910 * 3, 732
}

func main() {
	fmt.Println("It's starting!")
	ebiten.SetWindowSize(910*3/2, 732/2)
	ebiten.SetWindowTitle("Problem 1; Color Balance")
	ebiten.SetTPS(0)
	err := ebiten.RunGame(&Game{})
	if err != nil {
		log.Fatal(err)
	}
}
