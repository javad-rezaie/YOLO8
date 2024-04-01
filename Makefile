#---------------------HOMAI------------------------#
# Created on Sun Mar 31 2024
#
# Copyright (c) 2024 The Home Made AI (HOMAI)
# Author: Javad Rezaie
# License: Apache License 2.0
#---------------------HOMAI------------------------#

docker-build-mmyolo:
	DOCKER_BUILDKIT=1 docker build \
	-f docker/mmyolo.Dockerfile \
	-t mmyolo .
