#!/bin/bash
# nb. wid is in arcseconds and is the radius






nice -n 10 python stitchvd2_jan2015_apertures_sep.py J1218 1.00 lens
nice -n 10 python stitchvd2_may2014_apertures_sep.py J1218 1.5 source

nice -n 10 python stitchvd2_jan2015_apertures_sep.py J1144 1.00 lens
nice -n 10 python stitchvd2_may2014_apertures_sep.py J1144 1.5 source

nice -n 10 python stitchvd2_jan2015_apertures_sep.py J1125 1.00 lens
nice -n 10 python stitchvd2_may2014_apertures_sep.py J1125 1.5 source

nice -n 10 python stitchvd2_jan2015_apertures_sep.py J0913 1.00 lens
nice -n 10 python stitchvd2_may2014_apertures_sep.py J0913 1.5 source

nice -n 10 python stitchvd2_jan2015_apertures_sep.py J0901 1.00 lens
nice -n 10 python stitchvd2_may2014_apertures_sep.py J0901 1.5 source

nice -n 10 python stitchvd2_jan2015_apertures_sep.py J0837 1.00 lens
nice -n 10 python stitchvd2_may2014_apertures_sep.py J0837 1.5 source
