#!/bin/bash
d=/shared/home/namangoyal/checkpoints/175B
URL="https://fairacceleastus.blob.core.windows.net/$USER/logs_175B?sv=2020-08-04&ss=b&srt=sco&sp=rwdlactfx&se=2023-10-06T11:23:33Z&st=2021-10-06T03:23:33Z&spr=https&sig=s6aw4Ca4Ohbr7LQ%2BG9s58PEyYJsbXHjs%2Fc%2BuoTvzTUo%3D"
/shared/home/myleott/bin/azcopy_linux_amd64_10.12.1/azcopy cp --recursive --include-pattern "*.log"  $d $URL
