chelp+
c       Code to generate FT of final 2D-Gaussian to be used 
c       for convolving an image. The code deconvolves the input 
c       psf. The intrinsic psf must be specified. 
c                                          --wr, 22 Oct, 2020
c                                          Wasim.Raja@csiro.au
c
c INPUTS: 
c     bmin_in = intrinsic psf BMIN (degrees)
c     bmaj_in = intrinsic psf BMAJ (degrees)
c     bpa_in  = intrinsic psf BPA  (degrees)
c
c     bmin    = final psf BMIN (degrees)
c     bmaj    = final psf BMAJ (degrees)
c     bpa     = final psf BPA  (degrees)
c
c     u       = Fourier coordinates corresponding to image coord x 
c     v       = Fourier coordinates corresponding to image coord y
c
c     nx      = Number of pixels along x
c     ny      = Number of pixels along y
c
c OUTPUT: 
c     g_final = Final array to be multiplied to FT(image) to get teh
c               convoution in the FT domain.
c     g_ratio = factor for flux scaling
c
chelp-
      subroutine gaussft(bmin_in, bmaj_in, bpa_in, 
     -                    bmin, bmaj, bpa, 
     -                    u, v, nx, ny, 
     -                    g_final,g_ratio)
      implicit none 
      integer*4, intent(in) :: nx, ny 
      real*8, intent(out), dimension(nx,ny) :: g_final
      real*8, intent(out) :: g_ratio
      real*8, intent(in),dimension(nx) :: u 
      real*8, intent(in),dimension(ny) :: v 
      real*8, intent(in) :: bmin_in, bmaj_in, bpa_in
      real*8, intent(in) :: bmin, bmaj, bpa
      integer*4 ix, iy  

      ! Local variables
      real*8 pi
      real*8 cos_bpa, sin_bpa, cos_bpa_in, sin_bpa_in
      real*8 u_cosbpa(nx),u_sinbpa(nx),u_cosbpa_in(nx),u_sinbpa_in(nx) 
      real*8 v_cosbpa(ny),v_sinbpa(ny),v_cosbpa_in(ny),v_sinbpa_in(ny)
      real*8 g_amp, dg_amp 
      real*8 g_arg, dg_arg
      real*8 ur, vr 
      real*8 sx, sy, sx_in, sy_in
      real*8 deg2rad
      real*8 bmin_rad, bmaj_rad, bpa_rad 
      real*8 bmin_in_rad, bmaj_in_rad, bpa_in_rad 

      pi = acos(-1.0)
      deg2rad = pi/180.0
      !write(*,*)"pi: ",pi
      bmaj_in_rad = bmaj_in*deg2rad 
      bmin_in_rad = bmin_in*deg2rad 
      bpa_in_rad = bpa_in*deg2rad 

      bmaj_rad = bmaj*deg2rad 
      bmin_rad = bmin*deg2rad 
      bpa_rad = bpa*deg2rad 

      sx = bmaj_rad/(2*sqrt(2.0*log(2.0)))
      sy = bmin_rad/(2*sqrt(2.0*log(2.0)))
      sx_in = bmaj_in_rad/(2.0*sqrt(2.0*log(2.0)))
      sy_in = bmin_in_rad/(2.0*sqrt(2.0*log(2.0)))
      !write(*,*)"sx,sy,sx_in,sy_in: ",sx,sy,sx_in,sy_in

      do ix = 1,nx
        do iy = 1,ny
           g_final(ix,iy) = 0.0 
        enddo
      enddo 
      do ix=1,nx
         u_cosbpa(ix) = 0.0
         u_sinbpa(ix) = 0.0
         u_cosbpa_in(ix) = 0.0
         u_sinbpa_in(ix) = 0.0
      enddo
      do iy=1,ny
         v_cosbpa(iy) = 0.0
         v_sinbpa(iy) = 0.0
         v_cosbpa_in(iy) = 0.0
         v_sinbpa_in(iy) = 0.0
      enddo

      ! Store the constants to avoid unnecessary compute costs
      cos_bpa = cos(bpa_rad)
      sin_bpa = sin(bpa_rad)
      g_amp = sqrt(2.0*pi*sx*sy)

      cos_bpa_in = cos(bpa_in_rad)
      sin_bpa_in = sin(bpa_in_rad)
      dg_amp = sqrt(2.0*pi*sx_in*sy_in)

      g_ratio = g_amp/dg_amp 
      !write(77,*)g_ratio

      !write(*,*)"cbpa, sbpa, g_amp: ",cos_bpa,sin_bpa, g_amp
      !write(*,*)"gratio: ",g_ratio

      do ix = 1,nx 
          u_cosbpa(ix) = u(ix)*cos_bpa
          u_sinbpa(ix) = u(ix)*sin_bpa
    
          u_cosbpa_in(ix) = u(ix)*cos_bpa_in
          u_sinbpa_in(ix) = u(ix)*sin_bpa_in
      enddo
    
      do iy = 1,ny
          v_cosbpa(iy) = v(iy)*cos_bpa
          v_sinbpa(iy) = v(iy)*sin_bpa
    
          v_cosbpa_in(iy) = v(iy)*cos_bpa_in
          v_sinbpa_in(iy) = v(iy)*sin_bpa_in
      enddo
    
      do ix = 1,nx
         do iy = 1,ny
            ! Spectra of the Convolution kernel: 
        
            ur = u_cosbpa(ix) - v_sinbpa(iy)
            vr = u_sinbpa(ix) + v_cosbpa(iy)
            g_arg = -2.0*pi**2 * ((sx * ur)**2 + (sy * vr)**2)
        
        
            ! Spectra of the De-convolution kernel: 
            ur = u_cosbpa_in(ix) - v_sinbpa_in(iy)
            vr = u_sinbpa_in(ix) + v_cosbpa_in(iy)
            dg_arg = -2.0*pi**2 * ((sx_in*ur)**2 + (sy_in*vr)**2)
        
            g_final(ix,iy) = g_ratio * exp(g_arg - dg_arg)
         enddo
      enddo

      end
        
