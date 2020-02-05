import numpy as np
import copy

class RadTan(object):
    def __init__(self, calib, in_w, in_h, out_w, out_h):
        self.fx = calib[0]
        self.fy = calib[1]
        self.cx = calib[2]
        self.cy = calib[3]
        self.k1 = calib[4]
        self.k2 = calib[5]
        self.p1 = calib[6]
        self.p2 = calib[7]
        self.k3 = calib[8]

        self.in_w = in_h
        self.in_h = in_w

        self.out_w = out_w
        self.out_h = out_h

    def DistortCoor(self, in_x, in_y, n):
        out_x = []
        out_y = []

        for i in range(n):
            x = in_x[i]
            y = in_y[i]

            ix = (x - self.ocx) / self.ofx
            iy = (y - self.ocy) / self.ofy
            mx2_u = ix * ix
            my2_u = iy * iy
            mxy_u = ix * iy
            rho2_u = mx2_u + my2_u
            rad_dist_u = self.k1 * rho2_u + self.k2 * rho2_u * rho2_u + self.k3 * rho2_u * rho2_u * rho2_u
            x_dist = ix + ix * rad_dist_u + 2.0 * self.p1 * mxy_u + self.p2 * (rho2_u + 2.0 * mx2_u)
            y_dist = iy + iy * rad_dist_u + 2.0 * self.p2 * mxy_u + self.p1 * (rho2_u + 2.0 * my2_u)

            ox = self.fx * x_dist + self.cx
            oy = self.fy * y_dist + self.cy

            out_x.append(ox)
            out_y.append(oy)

        return out_x, out_y

    def MakeOptimalK_crop(self):
        # Finding CROP optimal new model
        self.K = np.eye(3)
        self.ofx = self.K[0, 0]
        self.ofy = self.K[1, 1]
        self.ocx = self.K[0, 2]
        self.ocy = self.K[1, 2]

        # 1. stretch the center lines as far as possible, to get initial coarse quess.
        tgX = []
        tgY = []
        minX = 0
        maxX = 0
        minY = 0
        maxY = 0

        for i in range(100000):
            tgX.append((i - 50000.0) / 10000.0)
            tgY.append(0.0)
        tgX, tgY = self.DistortCoor(tgX, tgY, 100000)

        for i in range(100000):
            if tgX[i] > 0 and tgX[i] < self.in_w - 1:
                if minX == 0:
                    minX = tgX[i]
                maxX = tgX[i]

        tmp = copy.deepcopy(tgX)
        tgX = copy.deepcopy(tgY)
        tgY = copy.deepcopy(tmp)

        for i in range(100000):
            if tgY[i] > 0 and tgY[i] < self.in_h - 1:
                if minY == 0:
                    minY = tgY[i]
                maxY = tgY[i]

        minX *= 1.01
        maxX *= 1.01
        minY *= 1.01
        maxY *= 1.01

        print('initial range: x: %.4f - %.4f; y: %.4f - %.4f!', minX, maxX, minY, maxY)

        # 2. while there are invalid pixels at the border: shrink square at the side that has invalid pixels,
        # if several to choose from, shrink the wider dimension.
        oobLeft = oobRight = oobTop = oobBottom = True
        iteration = 0
        remapX = []
        remapY = []
        while oobLeft or oobRight or oobTop or oobBottom:
            oobLeft = oobRight = oobTop = oobBottom = False
            for y in range(self.out_h):
                remapX.append(minX)
                remapX.append(maxX)
                tmp_y = minY + (maxY - minY) * float(y) / float(self.out_h - 1)
                remapY.append(tmp_y)
                remapY.append(tmp_y)

            remapX, remapY = self.DistortCoor(remapX, remapY, 2*self.out_h)

            for y in range(self.out_h):
                if (remapX[2*y] > 0 and remapX[2*y] < self.in_w - 1) is False:
                    oobLeft = True
                if (remapX[2*y + 1] > 0 and remapX[2*y + 1] < self.in_w -1) is False:
                    oobRight = True

            for x in range(self.out_w):
                remapY[2*x] = minY
                remapY[2*x + 1] = maxY
                tmp_x = minX + (maxX - minX) * float(x) / float(self.out_w - 1)
                remapX[2*x] = tmp_x
                remapX[2*x + 1] = tmp_x

            remapX, remapY = self.DistortCoor(remapX, remapY, 2*self.out_w)

            for x in range(self.out_w):
                if (remapY[2*x] > 0 and remapY[2*x] < self.in_h - 1) is False:
                    oobTop = True
                if (remapY[2*x + 1] > 0 and remapY[2*x + 1] < self.in_h -1) is False:
                    oobBottom = True

            if (oobLeft or oobRight) and (oobTop or oobBottom):
                if (maxX - minX) > (maxY - minY):
                    oobBottom = oobTop = False
                else:
                    oobLeft = oobRight = False

            if oobLeft:
                 minX *= 0.995
            if oobRight:
                 maxX *= 0.995
            if oobTop:
                minY *= 0.995
            if oobBottom:
                maxY *= 0.995

            iteration += 1

            if iteration > 500:
                print('FAILED TO COMPUTE GOOD CAMERA MATRIX - SOMETHING IS SERIOUSLY WRONG. ABORTING')

        self.K[0,0] = (self.out_w - 1.0) / (maxX - minX)
        self.K[1,1] = (self.out_h - 1.0) / (maxY - minY)
        self.K[0,2] = -minX * self.K[0,0]
        self.K[1,2] = -minY * self.K[1,1]