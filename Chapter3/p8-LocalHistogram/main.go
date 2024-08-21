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
var img3 *ebiten.Image
var img4 *ebiten.Image

func init() {
	var err error

	img, _, err = ebitenutil.NewImageFromFile("disney.png")
	if err != nil {
		panic(err)
	}

	img2 = ebiten.NewImage(800, 600)
	img3 = ebiten.NewImage(800, 600)
	img4 = ebiten.NewImage(800, 600)

}

func clamp64(x, vmin, vmax float64) (y float64) {
	var clamped float64
	if x < vmin {
		clamped = vmin
	} else if x > vmax {
		clamped = vmax
	} else {
		clamped = x
	}

	return clamped
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

func f(x float64) (y float64) {
	return math.Pow(x, 1.0/3.0)
	//	if x > math.Pow(6.0/29.0, 3) {
	//		return math.Pow(x, 1.0/3.0)
	//	}
	//
	// return x/(3.0*6.0*6.0/29.0/29.0) + 4.0/29.0
}

func finv(x float64) (y float64) {
	return math.Pow(x, 3.0)
	//	if x > 6.0/29.0 {
	//		return math.Pow(x, 3.0)
	//	}
	//
	// return 3 * (6.0 * 6.0 / 29.0 / 29.0) * (x - 4.0/29.0)
}

func extract_Lab(r, g, b float64) (Ls, as, bs float64) {
	X := (1 / 0.17697) * (0.49*r + 0.31*g + 0.2*b)
	Y := (1 / 0.17697) * (0.17697*r + 0.81240*g + 0.01063*b)
	Z := (1 / 0.17697) * (0.0*r + 0.01*g + 0.99*b)

	Lstar := (116*f(Y/100.0) - 16)
	astar := (500 * (f(X/95.0) - f(Y/100.0)))
	bstar := (200 * (f(Y/100.0) - f(Z/108.8)))

	return clamp64(Lstar, 0, 255), astar, bstar
}

func Lab_to_RGB(Ls, as, bs float64) (r, g, b uint8) {
	X := 95.0 * finv((Ls+16.0)/116.0+as/500.0)
	Y := 100.0 * finv((Ls+16.0)/116.0)
	Z := 108.8 * finv(((Ls+16.0)/116.0)-bs/200.0)

	R := 0.17697 * (3.24*X - 1.54*Y - 0.49*Z)
	G := 0.17697 * (-0.969*X + 1.876*Y + 0.04*Z)
	B := 0.17697 * (0.055*X - 0.020*Y + 1.057*Z)

	R = clamp64(R, 0, 255)
	G = clamp64(G, 0, 255)
	B = clamp64(B, 0, 255)

	return uint8(R), uint8(G), uint8(B)
	//return uint8(0.17697 * X), uint8(0.17697 * Y), uint8(0.17697 * Z)
}

type Game struct {
}

func (g *Game) Update() error {

	Larr := make([][]uint8, 800)
	for i := range Larr {
		Larr[i] = make([]uint8, 600)
	}
	subLarr := make([][]uint8, 100)
	for i := range subLarr {
		subLarr[i] = make([]uint8, 100)
	}

	aarr := make([][]float64, 800)
	for i := range aarr {
		aarr[i] = make([]float64, 600)
	}
	barr := make([][]float64, 800)
	for i := range barr {
		barr[i] = make([]float64, 600)
	}

	fmt.Println("Separating R, G, and B colors")
	for col := range 800 {
		for row := range 600 {
			c := img.At(col, row)
			r, g, b, _ := c.RGBA()
			rr, gg, bb := balancecolor(r, g, b, 1.0, 1.0)

			Ls, as, bs := extract_Lab(float64(rr), float64(gg), float64(bb))
			Larr[col][row] = uint8(Ls)
			aarr[col][row] = as
			barr[col][row] = bs

		}
	}

	//func cdf(ImgIn [][]uint8, rows, cols int) (_cdf [256]uint8) {
	for col := range 800 {
		for row := range 600 {
			c := img.At(col, row)
			r, g, b, _ := c.RGBA()
			rr, gg, bb := balancecolor(r, g, b, 1.0, 1.0)
			Ls, _, _ := extract_Lab(float64(rr), float64(gg), float64(bb))

			newc := color.Color(color.RGBA{uint8(Ls), uint8(Ls), uint8(Ls), 255})

			//newc := color.Color(color.RGBA{255, 255, 255, 255})
			img2.Set(col, row, newc)

		}
	}

	cdfs := make([][][]uint8, 8)
	for i := range cdfs {
		cdfs[i] = make([][]uint8, 6)
		for j := range cdfs[i] {
			cdfs[i][j] = make([]uint8, 256)

		}

	}

	for tilecol := range 8 {
		for tilerow := range 6 {
			fmt.Println(tilecol*4, (tilecol+1)*4)
			for col := range 100 {
				for row := range 100 {
					subLarr[col][row] = Larr[tilecol*100+col][tilerow*100+row]
				}
			}

			//			fmt.Println(Larr[tilecol*4 : (tilecol+1)*4][tilerow*4 : (tilerow+1)*4])
			mycdf := cdf(subLarr, 100, 100)
			cdfs[tilecol][tilerow] = mycdf[:]

		}
	}

	fmt.Println(cdfs)

	for tilecol := range 8 {
		for tilerow := range 6 {
			for col := range 100 {
				for row := range 100 {
					xind := row + tilecol*100
					yind := col + tilerow*100
					L := Larr[xind][yind]
					//L += 10 * (uint8(tilecol) + uint8(tilerow))

					newL := cdfs[tilecol][tilerow][L]

					newc := color.Color(color.RGBA{newL, newL, newL, 255})

					//newc := color.Color(color.RGBA{255, 255, 255, 255})
					img3.Set(xind, yind, newc)

				}
			}
		}
	}

	for tilecol := range 8 {
		for tilerow := range 6 {
			for col := range 100 {
				for row := range 100 {
					xind := row + tilecol*100
					yind := col + tilerow*100
					L := Larr[xind][yind]

					s := (float64(xind) - 50) / 100
					corrs := s - float64(int(s))
					if corrs < 0 {
						corrs = 0
					}
					t := (float64(yind) - 50) / 100
					corrt := t - float64(int(t))
					if corrt < 0 {
						corrt = 0
					}

					lefttileind := uint8((float64(xind) - 50) / 100)
					toptileind := uint8((float64(yind) - 50) / 100)

					tlcdf := cdfs[lefttileind][toptileind]
					trcdf := cdfs[(lefttileind+1)%8][toptileind]
					blcdf := cdfs[lefttileind][(toptileind+1)%6]
					brcdf := cdfs[(lefttileind+1)%8][(toptileind+1)%6]

					//leftcdf := cdfs[lefttileind][tilerow]
					//rightcdf := cdfs[(lefttileind+1)%8][tilerow]
					//topcdf := cdfs[][tilerow]
					//bottomcdf := cdfs[(lefttileind+1)%8][tilerow]
					tlL := tlcdf[L]
					trL := trcdf[L]
					blL := blcdf[L]
					brL := brcdf[L]

					//					newL := uint8((1-corrs)*float64(leftL) + corrs*float64(rightL))

					//newL := uint8((1-corrs)*float64(tlL) + corrs*float64(trL))
					//newL := uint8((1-corrt)*float64(tlL) + corrs*float64(trL))

					newL := uint8((1-corrs)*(1-corrt)*float64(tlL) +
						corrs*(1-corrt)*float64(trL) +
						(1-corrs)*corrt*float64(blL) +
						corrs*corrt*float64(brL))

					newc := color.Color(color.RGBA{newL, newL, newL, 255})

					//newc := color.Color(color.RGBA{255, 255, 255, 255})
					img4.Set(xind, yind, newc)

				}
			}
		}
	}

	fmt.Println("Building L* CDF")
	Lcdf := cdf(Larr, 800, 600)
	fmt.Println(Lcdf)

	fmt.Println("Updated!")
	return nil
}

func (g *Game) Draw(screen *ebiten.Image) {
	screen.DrawImage(img, nil)

	op := &ebiten.DrawImageOptions{}
	op.GeoM.Translate(800, 0)
	screen.DrawImage(img2, op)

	op = &ebiten.DrawImageOptions{}
	op.GeoM.Translate(0, 600)
	screen.DrawImage(img3, op)

	op = &ebiten.DrawImageOptions{}
	op.GeoM.Translate(800, 600)
	screen.DrawImage(img4, op)

}

func (g *Game) Layout(outsideWidth, outsideHeight int) (screenWidth, screenHeight int) {
	return 800 * 2, 600 * 2
}

func main() {
	fmt.Println("It's starting!")
	ebiten.SetWindowSize(1200, 900)
	ebiten.SetWindowTitle("Problem 1; Color Balance")
	ebiten.SetTPS(0)
	err := ebiten.RunGame(&Game{})
	if err != nil {
		log.Fatal(err)
	}
}
