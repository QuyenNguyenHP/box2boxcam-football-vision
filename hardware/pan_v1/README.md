# PAN V1 — đầu quay một trục cho iPhone 15 Pro Max

Thiết kế prototype ngày 25/11/2026. Hình học CAD được kiểm tra bằng phần mềm;
**chưa chế tạo, chưa thử tải hoặc xác nhận độ mượt ngoài sân**. Các kích thước vòng bi,
horn và tai servo là giao diện tham số cần xác nhận trước khi in trọn bộ.

## Phạm vi và cấu hình

- Điện thoại quay ngang, camera sau hướng ra sân; trục pan thẳng đứng.
- iPhone trần: 159,9 × 76,7 × 8,25 mm; 221 g. Có ốp cần đo lại. Độ dày thân
  này không bao gồm toàn bộ vùng lồi camera; CAD điện thoại chỉ là khối tham chiếu.
- Dự trù tải chuyển động 0,5 kg gồm điện thoại, ốp, kẹp, ốc và mâm; đây là mục tiêu
  thiết kế, không phải tải đã được chứng nhận. Dùng quả nặng giả trước khi gắn máy.
- Servo vị trí FEETECH FS5115M, truyền động đồng trục 1:1 qua horn.
- Vòng bi bàn xoay chịu tải và mômen lật; không treo điện thoại trực tiếp lên horn.
- Firmware ban đầu giới hạn ±60° quanh vị trí giữa, tốc độ 3–15°/s, gia tốc thử
  10–30°/s². Mở dần góc sau khi kiểm tra servo. Chặn cơ khí dự phòng khoảng ±70°
  theo hình học danh định; không dùng làm điểm dừng điều khiển thường xuyên.
- Đây là đầu pan có giới hạn góc, không quay liên tục 360°, không chống rung đa trục.
- Đặt ở gần giữa đường biên, máy cao và nhìn thoáng sân. Góc cần thiết phụ thuộc khoảng
  cách đến sân; 120° chuyển động cộng góc nhìn camera chưa bảo đảm bao trọn mọi vị trí.

## Cấu trúc và đường truyền tải

```text
iPhone nằm ngang + kẹp kim loại ren cái 1/4-20
                 ↓ vít 1/4-20 qua rãnh trên mâm
mâm PETG Ø110, dày 6; tai chặn vươn bán kính 66
                 ↓ bắt vào lá quay của vòng bi
vòng bi bàn xoay vuông ~76,2 mm, lỗ giữa >=46 mm
                 ↓ lá cố định bắt vào nắp PETG dày 6
hộp 150 × 140 × 72 → đai ốc kim loại 1/4-20 → tripod

servo cố định bên dưới nắp → horn → spacer dưới mâm → quay mâm
```

Không dùng lực siết horn để kéo mâm xuống hoặc nâng mâm lên khỏi vòng bi. Điều chỉnh
cao độ khay servo bằng 4 thanh ren M3, êcu kép và shim sao cho horn vừa chạm spacer.
Vòng bi và servo phải đồng tâm; kiểm tra quay nhẹ khi tháo horn trước khi nối truyền động.
Lắp cứng lệch tâm có thể làm kẹt và rung; nếu không căn được, phải đổi sang khớp nối
cho phép lệch nhỏ hoặc truyền đai, thay vì tăng mômen servo để ép chạy.

Tâm ren kẹp lệch +22 mm theo Y để tránh vít tâm horn. Dùng rãnh X ±16 mm chỉnh
cân bằng ngang; chọn kẹp cho phép đưa tâm khối lượng máy gần trục pan. Offset Y còn lại
tạo mômen lật, đã được giao cho vòng bi đỡ; không giả định trọng tâm hoàn toàn đồng trục.

### Kích thước CAD danh định

| Chi tiết | Kích thước / cao độ lắp (mm) |
|---|---|
| Hộp | 150 × 140 × 72; đáy 9; thành 3 |
| Nắp | Z=72…78; tấm 6; chốt dự phòng lên đến Z=93 |
| Vòng bi tham chiếu | 76,2 × 76,2 × 10; Z=78…88; tâm rỗng >=46 |
| Lỗ bắt vòng bi | 4 lỗ Ø4,5; bước vuông 58 × 58 — giả định cần đo |
| Mâm | Ø110; Z=88…94; spacer dưới mâm Z=84…88 |
| Khay servo | 112 × 64 × 5; mặt trên Z=64, thay đổi bằng êcu |
| Lỗ thân servo | 42 × 22; tâm X=+10 so với trục quay |
| Tai servo | bước X=49,5; Y=10, rãnh ±1 theo X — cần đo |
| Horn | Ø<=36; lỗ truyền động ±12 theo X — cần đo |
| Lỗ kẹp | rãnh rộng 6,9; tâm Y=+22, X chạy ±16 |
| Giao diện tripod | đai ốc 1/4-20 kim loại; không in ren chịu tải |

Tổng từ đáy hộp đến mặt mâm là 94 mm với vòng bi danh định 10 mm. Chiều cao toàn
bộ còn phụ thuộc kẹp mua được; khối điện thoại trong preview có vị trí minh họa.

## Danh sách mua

Xem [BOM.csv](BOM.csv): có số lượng, thông số chọn hàng, ốc và nguồn. Đồ tham chiếu:

- [FEETECH FS5115M](https://www.pololu.com/product/3426/specs): nguồn 4,8–6 V,
  tương thích tín hiệu 3,3 V; chọn đúng servo vị trí. Khác revision có thể khác tai lắp.
- [Vòng bi ServoCity 3 inch, 1620-0001-0003](https://www.servocity.com/3x3-ball-bearing-turntable/):
  ứng viên tham khảo. **Chưa xác minh lỗ tâm, bước lỗ và chiều dày của mã này**;
  chỉ mua sau khi người bán cung cấp bản vẽ đáp ứng giao diện hoặc chỉnh CAD theo bản vẽ.
- Kẹp kim loại mở 70–90 mm hoặc rộng hơn; ren cái 1/4-20 ở đáy; đệm cao su;
  vít siết cơ khí, không dùng MagSafe đơn lẻ. Không ép nút nguồn/âm lượng hoặc che camera.
- ESP32 DevKitC V4 dùng module WROOM-32E, có BLE. Hộp dùng đệm cách điện và dây
  rút nên không phụ thuộc bước lỗ PCB.

Không chọn servo quay liên tục hoặc servo ghi “360° continuous”. Không thay servo khác
chỉ dựa vào nhãn “20 kg”: cần biết điện áp, loại điều khiển, spline và bản vẽ kích thước.
Servo hobby có độ rơ và deadband; độ mượt video cần đo thực tế, không được suy ra từ mômen.

## Nguồn và dây điện

```text
Nguồn DC 12 V ngoài hộp
  → cầu chì phù hợp dây + công tắc DC
      ├── buck 6,0 V → servo (+) ; tụ 1000 µF/10 V gần servo
      └── buck 5,0 V → chân 5V/VIN ESP32 theo đúng sơ đồ board
GND nguồn / 2 buck / servo / ESP32 nối chung kiểu sao
ESP32 GPIO18 → điện trở 220–470 Ω → signal servo
```

Đặt buck đúng điện áp bằng đồng hồ trước khi nối tải. Nhánh servo ưu tiên module có
khả năng liên tục >=3 A và đỉnh >=5 A, phải kiểm chứng nhiệt và dòng thực; không lấy
nguồn servo từ ESP32. Không chọn nguồn chỉ dựa vào con số quảng cáo trên module.
Đề xuất thử bàn bằng adapter DC12V/3A có bảo vệ; ngoài sân đổi sang bộ pin đóng gói có
BMS và bộ sạc phù hợp. Hộp không chứa cell lithium rời. Không dùng USB5V power bank
cắm trực tiếp vào nhánh6V của thiết kế này. Không cấp đồng thời USB và5V ngoài cho ESP32
nếu board không có mạch chọn nguồn phù hợp. Điện thoại dùng pin riêng ở bản đầu.

Đầu cáp hộp cần grommet và chống kéo; lỗ chữ nhật14×10 chỉ là cổng cáp danh định,
không phải lỗ gland chuẩn có ren. Hộp hiện không có gasket và không chống nước.

## File 3D, tái tạo và in

- `exports/base.stl`, `lid.stl`, `servo_tray.stl`, `rotor.stl`: bốn chi tiết cần in.
- `exports/fit_coupon.stl`: mẫu thử kích thước trước khi in đủ.
- `exports/assembly.glb`: xem bố trí 3D, bao gồm các khối tham chiếu linh kiện mua sẵn.
- `exports/exploded_preview.png`: ảnh tháo rời minh họa; khoảng cách tháo rời không phải
  khoảng cách lắp. Khối vòng bi, servo, kẹp và điện thoại trong GLB không phải chi tiết để in.
- `exports/dimensions.png`: sơ đồ kích thước mặt bằng và cao độ lắp.
- `exports/geometry_check.json`: kết quả kiểm tra kín mặt và thể tích.
  Kiểm tra mâm quay không giao nhau với các chi tiết in cố định ở các góc cách nhau5°
  trong phạm vi−60°…+60°; chưa bao gồm ốc, dung sai in hoặc biến dạng khi chịu tải.
- `build_cad.py`: CAD tham số bằng Python/trimesh/manifold; đơn vị mm. Một số kích
  thước hộp cố định trong hàm tạo khối; đổi form hộp cần sửa chúng cùng các trường P.

```bash
python3 -m venv .venv-cad
source .venv-cad/bin/activate
pip install -r hardware/pan_v1/requirements-cad.txt
python hardware/pan_v1/build_cad.py
```

PETG là vật liệu đề xuất; đầu pan ngoài nắng không nên thử bằng PLA mỏng. Khởi điểm:
nozzle0,4mm, layer0,2mm, 5–6 vòng thành, 6 lớp trên/dưới, infill35–45%; tùy máy in
cần hiệu chỉnh co ngót và lỗ. In nắp đáy phẳng xuống, post hướng lên. Mâm in với mặt
trên áp bàn, spacer hướng lên (xoay180° trong slicer); khay và hộp đáy xuống. STL đã
đưa đáy về Z=0, nhưng người in vẫn phải chọn đúng mặt đặt. Coupon và khay không cần
support; xem slicer cho lỗ ngang công tắc/cáp ở hộp, có thể cần support cục bộ.

Chừa 0,2–0,4mm khe hở lắp ghép là điểm bắt đầu, không phải đảm bảo độ chính xác.
Lỗ ốc có thể khoan rà; không ép linh kiện vào lỗ chật hoặc tăng lực siết làm nứt PETG.
Đai ốc tripod nằm trong túi lục giác đáy; giữ bằng epoxy sau kiểm tra, giữ sạch ren.

## Trình tự lắp và kiểm tra

1. Đo các giao diện trong checklist bên dưới; sửa P và xuất lại nếu khác.
2. In coupon, thử thân servo và các lỗ ốc; chỉnh bù lỗ theo máy in.
3. In đủ; làm sạch ba via. Lắp êcu M3 nắp vào hốc trên 4 trụ hộp, giữ chống rơi.
4. Lắp êcu1/4-20 vào đáy, kiểm tra vít tripod không chạm điện tử. Lót cách điện PCB.
5. Thử servo riêng tại1500µs; đánh dấu hướng giữa. Chỉ mở biên xung dần khi không tải;
   không mặc định500–2500µs an toàn cho mọi servo/clone.
6. Lắp servo lên khay, treo khay dưới nắp bằng4 thanh ren. Chưa nối horn với mâm.
7. Bắt lá cố định vòng bi vào nắp. Lắp horn và vít tâm đúng spline. Bắt mâm vào lá quay,
   xoay vòng bi để tiếp cận từng lỗ nếu cấu tạo cho phép; không để đầu ốc cạ lá còn lại.
8. Tháo mâm nếu cần để đặt vít nối horn; nâng/hạ khay cho horn chạm spacer tự nhiên,
   không làm mất độ quay nhẹ của vòng bi. Bắt2 vít horn, kiểm tra lại và khóa êcu khay.
9. Lắp nắp lên hộp. Vít khay đầu chìm/đầu thấp phải không chạm lá vòng bi; xác nhận
   kích thước vòng bi thực không đè lên các vít khay tại X±45,Y±23.
10. Bắt kẹp vào rãnh mâm bằng vít1/4-20 có chiều dài đúng; bảo đảm không đụng vít horn,
    camera và nút bấm. Gắn dây giữ điện thoại vào phần quay. Lắp tripod và khóa góc nghiêng.
11. Thử tải giả0,5kg thấp và chắc: đứng yên, xoay chậm±30°, rồi±60°. Kiểm tra nứt,
    rung, độ rơ, tụt nguồn, nóng servo/buck và chân tripod. Không test bằng máy thật trước.
12. Kiểm tra watchdog mất lệnh300–500ms: ngừng cập nhật góc đích, giữ vị trí hiện tại.
    Với servo vị trí, “dừng” vẫn giữ lực; tắt xung không tương đương phanh và có thể thả tải.
13. Gắn máy thật và quay video ngắn để đánh giá hình; sau đó mới ghép detection tự động.

Chốt nắp là chặn dự phòng, gắn ống cao su và bu-lôngM4 nếu cần. Bề rộng tai và post
làm góc chạm nhỏ hơn80°; danh định khoảng70°. **Đo lại góc chạm thực** và để firmware
dừng trước đó ít nhất5–10°. Không cấp lệnh cố ép servo lên chặn.

## Checklist trước khi in sản xuất

| Đo thực tế | Dùng để chỉnh |
|---|---|
| Vòng bi: cạnh, cao, lỗ tâm, bước lỗ cả2 lá, đầu ốc | `bearing_height`, `bearing_pitch`, `bearing_hole`; sửa envelope nếu cần |
| Có đủ lối bắt vít độc lập hai lá không? | Trình tự lắp / lỗ tiếp cận bổ sung |
| Servo: thân, khoảng tai, vị trí trục so với tâm thân | `servo_body_*`, `servo_ear_pitch_*` |
| Cao từ tai servo đến mặt trên horn đã bắt vít | Chỉnh khay M3; `tray_top`, `horn_top` và spacer |
| Horn: spline, đường kính, bước2 lỗ, ren vít | `horn_pitch`, spacer và vít |
| Kẹp: ren đáy, bề rộng mở, cao, nút/camera bị che không | Chiều dài vít và vị trí kẹp |
| Điện thoại có ốp: bề rộng thật và tổng khối lượng | Kẹp và thử tải |

Không có dữ liệu lực/nhiệt đo từ mẫu thật nên chưa tính độ bền chứng nhận. Với tải
giả0,5kg lệch trục22mm, mômen lật tĩnh xấp xỉ0,108N·m; đây là tải cần vòng bi/nắp
chịu, **không phải** mômen servo pan cần. Servo pan thắng quán tính, ma sát, gió và
dây kéo; chọn/mua dựa thêm vào độ rơ và thử thực tế.

## Nguồn tham khảo

- [Apple: kích thước iPhone15ProMax](https://support.apple.com/en-us/111828).
- [Pololu: servo FS5115M](https://www.pololu.com/product/3426/specs).
- [Datasheet FEETECH, revision2021](https://www.pololu.com/file/0J1270/FS5115M-specs.pdf).
- [ServoCity: vòng bi3inch ứng viên](https://www.servocity.com/3x3-ball-bearing-turntable/).
- [Espressif: ESP32 DevKitC V4](https://documentation.espressif.com/esp-dev-kits/en/latest/esp32/esp32-devkitc/user_guide.html).

Thiết kế này chưa gồm firmware ESP32, giao thức BLE hoàn chỉnh hoặc chỉnh sửa app iOS.
