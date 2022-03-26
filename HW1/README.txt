#輸入僅限jpg檔
步驟1: 	將所有不同曝光時間之圖片放入同一個資料夾，並將檔名命名為img01.jpg,img02.jpg,img03.jpg…
步驟2: 	在此資料夾中創建time.data檔案，檔案中每行包含一個浮點數(或是1/數字)，該數字對應到圖片的曝光時間
步驟3: 	開啟cmd/terminal，輸入以下指令，根據所需調整參數
	HW1.exe [圖片數量] [圖片資料夾] [圖片附檔名] [-MTB] [-Debevec/-Robertson] [-global/-local/-bilateral/-logarithjmic]
	Ps. 只有圖片數量與圖片資料夾與附檔名是必須輸入的，其餘皆有預設值，可不輸入
	EX1: HW1.exe 10 ./images -MTB -local
	EX2: HW1.exe 13 ./exposures -bilateral
步驟4: 執行程式並等待!

編譯方式(適用於windows):
開啟./code/HW1_VisualStudio/HW1.sln
方案組態設為Release即可執行

程式碼dependency:
opencv2 https://opencv.org/
cvplot https://github.com/Profactor/cv-plot
eigen https://eigen.tuxfamily.org/index.php?title=Main_Page
tinyexpr https://github.com/codeplea/tinyexpr