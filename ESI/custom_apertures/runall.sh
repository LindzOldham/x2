#!/bin/bash
# nb. wid is in arcseconds and is the radius

nice -n 10 python extract_J1347_onesourceap.py 1.5
nice -n 10 python extract_J1323_onesourceap.py 1.5
nice -n 10 python extract_J1446_onesourceap.py 1.5
nice -n 10 python extract_J1605_onesourceap.py 1.5
nice -n 10 python extract_J1606_onesourceap.py 1.5
nice -n 10 python extract_J1619_onesourceap.py 1.5
nice -n 10 python extract_J2228_onesourceap.py 1.5



