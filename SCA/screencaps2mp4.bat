delete out.mp4
ffmpeg -r 29.970 -f image2 -s 1280x720 -i Screenshots\frame%%06d.png -vcodec libx264 -pix_fmt yuv420p -b:v 20000k out.mp4
