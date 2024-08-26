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

func spline(x float64) (y float64) {
	a := -0.5
	if x < 1 {
		return 1 - (a+3)*math.Pow(x, 2) + (a+2)*math.Pow(x, 3)
	} else if x < 2 {
		return a * (x - 1) * (x - 2) * (x - 2)
	} else {
		return 0
	}
}

func interp(inpImg *ebiten.Image) (outImg *ebiten.Image) {

	p := inpImg.Bounds().Size()
	w := p.X
	h := p.Y
	r := 2

	highResImg := ebiten.NewImage(r*w, r*h)

	for row := range r * w {
		for col := range r * h {
			//			frow := float64(row)
			//			fcol := float64(col)
			finalr := float64(0)
			totkern := float64(0)
			//			finalb := int32(0)
			//			finalg := int32(0)
			//			fmt.Println("For Pixel ", row, col)
			for ii := -3; ii < 4; ii++ {
				for jj := -3; jj < 4; jj++ {
					c := inpImg.At(((row/r+ii)+w)%w, ((col/r+jj)+h)%h)
					red, _, _, _ := c.RGBA()
					dx := float64(ii) / float64(r)
					dy := float64(jj) / float64(r)
					dist := math.Sqrt(math.Pow(dx, 2) + math.Pow(dy, 2))
					//					fmt.Println(dx, dy, dist, spline(dist))
					sval := spline(dist)
					finalr += sval * float64(red)
					totkern += sval

					//					finalg += kernel[ii][jj] * int32(g/256)
					//					finalb += kernel[ii][jj] * int32(b/256)
				}

			}
			ifinalr := int32(finalr / totkern / 256)
			//			finalg /= int32(256)
			//			finalb /= int32(256)
			//
			newc := color.Color(color.RGBA{uint8(clamp(ifinalr, 0, 255)),
				uint8(clamp(ifinalr, 0, 255)),
				uint8(clamp(ifinalr, 0, 255)),
				255})
			highResImg.Set(row, col, newc)
		}
	}

	return highResImg
}

func decimate(inpImg *ebiten.Image) (outImg *ebiten.Image) {

	p := inpImg.Bounds().Size()
	w := p.X
	h := p.Y
	r := 10

	kernel := [5][5]int32{
		{1, 4, 6, 4, 1},
		{4, 16, 24, 16, 4},
		{6, 24, 36, 24, 6},
		{4, 16, 24, 16, 4},
		{1, 4, 6, 4, 1},
	}

	lowResImg := ebiten.NewImage(w/r, h/r)

	for row := range w / r {
		for col := range h / r {
			//			frow := float64(row)
			//			fcol := float64(col)
			finalr := float64(0)
			totkern := float64(0)
			//			finalb := int32(0)
			//			finalg := int32(0)
			//			fmt.Println("For Pixel ", row, col)
			for ii := -2; ii < 3; ii++ {
				for jj := -2; jj < 3; jj++ {
					c := inpImg.At(((r*row+ii)+w)%w, ((r*col+jj)+h)%h)
					red, _, _, _ := c.RGBA()
					//					fmt.Println(dx, dy, dist, spline(dist))
					sval := kernel[ii+2][jj+2]
					finalr += float64(sval) * float64(red)
					totkern += float64(sval)

					//					finalg += kernel[ii][jj] * int32(g/256)
					//					finalb += kernel[ii][jj] * int32(b/256)
				}

			}
			ifinalr := int32(finalr / totkern / 256)
			//			finalg /= int32(256)
			//			finalb /= int32(256)
			//
			newc := color.Color(color.RGBA{uint8(clamp(ifinalr, 0, 255)),
				uint8(clamp(ifinalr, 0, 255)),
				uint8(clamp(ifinalr, 0, 255)),
				255})
			lowResImg.Set(row, col, newc)
		}
	}

	return lowResImg
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

type Game struct {
	biggerImg  *ebiten.Image
	smallerImg *ebiten.Image
}

func (g *Game) Update() error {
	g.biggerImg = interp(img)
	g.smallerImg = decimate(img)

	fmt.Println("Updated!")
	return nil
}

func (g *Game) Draw(screen *ebiten.Image) {

	screen.DrawImage(img, nil)

	op := &ebiten.DrawImageOptions{}
	op.GeoM.Scale(0.5, 0.5)
	op.GeoM.Translate(320, 0)

	screen.DrawImage(g.biggerImg, op)

	op = &ebiten.DrawImageOptions{}
	op.GeoM.Translate(640, 0)
	screen.DrawImage(g.smallerImg, op)

}

func (g *Game) Layout(outsideWidth, outsideHeight int) (screenWidth, screenHeight int) {

	return 320 * 3, 320
}

func main() {
	fmt.Println("It's starting!")
	ebiten.SetWindowSize(320*6, 320*2)
	ebiten.SetWindowTitle("Problem 12; Bilateral Filtering")
	ebiten.SetTPS(0)
	err := ebiten.RunGame(&Game{})
	if err != nil {
		log.Fatal(err)
	}
}
