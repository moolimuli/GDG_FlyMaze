驗證地圖組（Stage A）：只用來評估，不可用於示範、訓練或調參數。
timeout = ceil(最短路線 mm / 7) + 2 秒。S=起點 F=食物 *=最短路線 o=岔路/死路 .=牆

custom_val_zigzag  route 110 mm  timeout 18 s  — 階梯形，左右交替轉 5 次
y5 . . . . . . F
y4 . . . . * * *
y3 . . . . * . .
y2 . . * * * . .
y1 . . * . . . .
y0 S * * . . . .
   0 1 2 3 4 5 6

custom_val_spiral  route 140 mm  timeout 22 s  — 螺旋，連續同方向（左）轉 4 次
y4 . * * * *
y3 . * . . *
y2 . * F . *
y1 . . . . *
y0 S * * * *
   0 1 2 3 4

custom_val_uturn  route 100 mm  timeout 17 s  — U 形迴轉，食物直線距離近但要繞路
y2 F * * * *
y1 . . . . *
y0 S * * * *
   0 1 2 3 4

custom_val_tjunction  route 70 mm  timeout 12 s  — 對稱 T 字岔路，一邊是死路
y5 . . . * * F
y4 . . . * . .
y3 S * * * . .
y2 . . . o . .
y1 . . . o o o
y0 . . . . . .
   0 1 2 3 4 5

custom_val_trap  route 130 mm  timeout 21 s  — 死路岔路的盡頭貼近食物
y4 . . * * * *
y3 . . F . . *
y2 . . . o o *
y1 . . . . . *
y0 S * * * * *
   0 1 2 3 4 5

custom_val_longrun  route 140 mm  timeout 22 s  — 長直線後兩個轉彎
y3 . . . . . . . F * *
y2 . . . . . . . . . *
y1 . . . . . . . . . *
y0 S * * * * * * * * *
   0 1 2 3 4 5 6 7 8 9

custom_val_comb  route 100 mm  timeout 17 s  — 主走廊旁有三條短死路
y3 . . o . . . o . F
y2 . . o . . . o . *
y1 S * * * * * * * *
y0 . . . . o . . . .
   0 1 2 3 4 5 6 7 8

custom_val_loop  route 90 mm  timeout 15 s  — 環狀路線（兩條路）再接出口
y4 . . . * * F
y3 . . . * . .
y2 . . * * o .
y1 . . * . o .
y0 S * * o o .
   0 1 2 3 4 5

