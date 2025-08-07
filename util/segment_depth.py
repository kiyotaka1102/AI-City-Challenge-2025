depth_file = h5py.File('/kaggle/input/depth-file/zip/depth_maps/Camera_01.h5', 'r')


frame_ids = sorted(frame_to_objects.keys())[:8000]


for idx, frame_id in enumerate(frame_ids):
    depth_key = f"distance_to_image_plane_{frame_id:05d}.png"
    if depth_key not in depth_file:
        print(f"⚠️ Key '{depth_key}' not found.")
        continue


depth_map = depth_file[depth_key][:]



for idx, frame_id in enumerate(frame_ids):
    depth_key = f"distance_to_image_plane_{frame_id:05d}.png"
    if depth_key not in depth_file:
        print(f"⚠️ Key '{depth_key}' not found.")
        continue

    depth_map = depth_file[depth_key][:]  # (H, W) mm

    for obj in frame_to_objects[frame_id]:
        bbox = obj['bbox_2d']
        x1, y1, x2, y2 = int(bbox['x1']), int(bbox['y1']), int(bbox['x2']), int(bbox['y2'])
        x1, y1, x2, y2 = x1 - 10, y1 - 10, x2 + 10, y2 + 10

        # === Crop bbox & threshold ===
        cropped_depth = depth_map[y1:y2, x1:x2]
        thresh_point = (x2, y2)

        # Kiểm tra không vượt ảnh
        if thresh_point[1] >= depth_map.shape[0] or thresh_point[0] >= depth_map.shape[1]:
            continue

        thresh = depth_map[thresh_point[1], thresh_point[0]]  # mm

        # === Mask từ threshold ===
        mask = (cropped_depth > 0) & (cropped_depth < thresh)
        mask = mask.astype(np.uint8) * 255

        # === Dùng minAreaRect tìm hướng object (nếu có mask) ===
        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        rect_vis = np.copy(mask)
        if contours:
            largest = max(contours, key=cv.contourArea)
            rect = cv.minAreaRect(largest)
            center = (int(rect[0][0]), int(rect[0][1]))  # rect[0] là center
            center_crop = (int(rect[0][0]), int(rect[0][1]))  # trong bbox crop
            center_full = (x1 + center_crop[0], y1 + center_crop[1])  # chuyển về ảnh gốc
            
            print(f"📍 Center in full depth image: {center_full}")

            box = cv.boxPoints(rect)
            box = np.intp(box)
            cv.drawContours(rect_vis, [box], 0, 255, 2)

        # === Vẽ điểm threshold lên ảnh depth full ===
        depth_vis = cv.normalize(depth_map, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
        depth_vis = cv.cvtColor(depth_vis, cv.COLOR_GRAY2BGR)
        cv.circle(depth_vis, thresh_point, 5, (0, 0, 255), -1)
        cv.circle(depth_vis, center_full, 5, (0, 255, 255), -1)  # màu vàng
        cv.rectangle(depth_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # === Hiển thị mỗi 100 frame ===
        if idx % 100 == 0:
            plt.figure(figsize=(15, 4))

            plt.subplot(1, 3, 1)
            plt.title(f"Depth w/ threshold point {frame_id}")
            plt.imshow(depth_vis[..., ::-1])
            plt.axis('off')

            plt.subplot(1, 3, 2)
            plt.title("Mask")
            plt.imshow(mask, cmap='gray')
            plt.axis('off')

            plt.subplot(1, 3, 3)
            plt.title("MinAreaRect on Mask")
            plt.imshow(rect_vis, cmap='gray')
            plt.axis('off')

            plt.tight_layout()
            plt.show()

depth_file.close()