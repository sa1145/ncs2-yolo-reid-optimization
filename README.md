# ncs2-yolo-reid-optimization

本repo為自走車人類跟隨系統專題中，視覺辨識模組（YOLO11n + ReID）於NCS2邊緣裝置之效能測試與優化記錄。完整系統程式碼由團隊共同維護，此處呈現本人負責之非同步運算資源分配決策（YOLO在NCS2執行、ReID在CPU執行）與相關實測數據。

## 測試結果摘要
YOLO11n於NCS2 Async×3：30.72 FPS（vs CPU opset10 17.16 FPS）
ReID於CPU：194.85 samples/s（vs NCS2 Async×4 91.75 samples/s）

結論：裝置分工（YOLO→NCS2、ReID→CPU）較能穩定的分配效能與兼顧速度