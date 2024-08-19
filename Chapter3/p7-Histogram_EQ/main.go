package main

import (
	"fmt"
	"image/color"
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

func cdf(ImgIn [][]uint8, rows, cols int) (_cdf [256]uint8) {
	var cdf [256]uint8
	var Ntot uint32 = 0

	for ival := range 256 {
		Ntot = 1
		for ir := range rows {
			for ic := range cols {

				if ImgIn[ir][ic] < uint8(ival) {
					Ntot += 1
				}

			}
		}
		cdf[ival] = uint8(256 * Ntot / uint32(rows) / uint32(cols))
	}
	return cdf
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

	rs := make([][]uint8, 858)
	for i := range rs {
		rs[i] = make([]uint8, 642)
	}
	gs := make([][]uint8, 858)
	for i := range gs {
		gs[i] = make([]uint8, 642)
	}
	bs := make([][]uint8, 858)
	for i := range bs {
		bs[i] = make([]uint8, 642)
	}
	fmt.Println("Separating R, G, and B colors")
	for col := range 858 {
		for row := range 642 {
			c := img.At(col, row)
			r, g, b, _ := c.RGBA()
			rr, gg, bb := balancecolor(r, g, b, 1.0, 1.0)

			rs[col][row] = rr
			gs[col][row] = gg
			bs[col][row] = bb

		}
	}

	//func cdf(ImgIn [][]uint8, rows, cols int) (_cdf [256]uint8) {
	fmt.Println("Building R CDF")
	rcdf := cdf(rs, 858, 642)
	fmt.Println("Building G CDF")
	gcdf := cdf(gs, 858, 642)
	fmt.Println("Building B CDF")
	bcdf := cdf(bs, 858, 642)

	for col := range 858 {
		for row := range 642 {
			c := img.At(col, row)
			r, g, b, _ := c.RGBA()
			rr, gg, bb := balancecolor(r, g, b, 1.0, 1.0)

			newc := color.Color(color.RGBA{rcdf[rr], gcdf[gg], bcdf[bb], 255})

			//newc := color.Color(color.RGBA{255, 255, 255, 255})
			img2.Set(col, row, newc)

		}
	}

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
