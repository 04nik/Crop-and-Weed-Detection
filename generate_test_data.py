# generate_test_data.py
import numpy as np
import cv2
import random
import os

def generate_field_image(width=800, height=600, num_rows=3, weed_density=15, output_path="field_sample.jpg"):
    # Create soil background (brownish color, e.g. H: 15, S: 50%, V: 30%)
    # In BGR: [35, 45, 60] roughly
    bg_color = [35, 48, 65]
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:, :] = bg_color

    # Add soil texture/noise
    noise = np.random.normal(0, 8, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    # Add some random soil clods/pebbles (grey/dark brown circles)
    for _ in range(150):
        cx = random.randint(0, width - 1)
        cy = random.randint(0, height - 1)
        r = random.randint(2, 6)
        c = random.randint(20, 40)
        cv2.circle(img, (cx, cy), r, (c, c+5, c+10), -1)

    # 1. Draw Crop Rows
    # Crops are aligned in vertical rows. Let's space them.
    row_xs = []
    margin = width // (num_rows + 1)
    for i in range(num_rows):
        row_xs.append((i + 1) * margin)

    crop_color = (40, 185, 80) # BGR: green
    crop_shadow_color = (25, 120, 50)

    # Draw crops along vertical lines with some spacing and randomness
    for x in row_xs:
        y_step = 60
        y = 40
        while y < height - 20:
            cx = x + random.randint(-8, 8)
            cy = y + random.randint(-5, 5)
            # Size of the crop plant
            size = random.randint(22, 28)
            
            # Draw individual leaves as ellipses radiating from center
            num_leaves = random.randint(4, 6)
            for j in range(num_leaves):
                angle = (j * 360 / num_leaves) + random.randint(-15, 15)
                # Leaf dimensions
                major = size
                minor = random.randint(10, 14)
                
                # Shadow
                cv2.ellipse(img, (cx, cy), (major, minor), angle, 0, 360, crop_shadow_color, -1)
                # Main Leaf
                cv2.ellipse(img, (cx, cy), (int(major*0.95), int(minor*0.95)), angle, 0, 360, crop_color, -1)
                # Leaf vein (lighter line)
                rad = np.deg2rad(angle)
                vx = int(cx + np.cos(rad) * major * 0.7)
                vy = int(cy + np.sin(rad) * major * 0.7)
                cv2.line(img, (cx, cy), (vx, vy), (70, 210, 110), 1)

            y += y_step + random.randint(-8, 8)

    # 2. Draw Weeds
    # Weeds are scattered randomly, but mostly between rows
    weed_color = (40, 130, 90) # BGR: Olive-green/darker green
    weed_shadow_color = (20, 80, 50)
    
    # We will place weeds randomly
    for _ in range(weed_density):
        # We try to keep weeds mostly away from crop centers to make visual separation cleaner, 
        # but some overlap is fine
        wx = random.randint(20, width - 20)
        wy = random.randint(20, height - 20)
        
        # Decide weed shape: spiky cluster or small group of small round circles
        weed_type = random.choice(["spiky", "round_clump"])
        
        if weed_type == "spiky":
            # Spiky leaves: draw thin, long ellipses (like crabgrass)
            num_spikes = random.randint(5, 8)
            size = random.randint(10, 15)
            for j in range(num_spikes):
                angle = j * 360 / num_spikes + random.randint(-10, 10)
                cv2.ellipse(img, (wx, wy), (size, random.randint(2, 4)), angle, 0, 360, weed_shadow_color, -1)
                cv2.ellipse(img, (wx, wy), (size-1, random.randint(2, 3)), angle, 0, 360, weed_color, -1)
        else:
            # Round clump: draw 3-4 small circles clumped together (like clover)
            num_clumps = random.randint(3, 5)
            for j in range(num_clumps):
                offset_x = random.randint(-6, 6)
                offset_y = random.randint(-6, 6)
                r = random.randint(4, 7)
                cv2.circle(img, (wx + offset_x, wy + offset_y), r, weed_shadow_color, -1)
                cv2.circle(img, (wx + offset_x, wy + offset_y), r - 1, weed_color, -1)

    cv2.imwrite(output_path, img)
    print(f"Generated synthetic field image: {output_path} ({width}x{height})")

if __name__ == "__main__":
    os.makedirs("test_data", exist_ok=True)
    # Generate a standard test field image
    generate_field_image(800, 600, num_rows=3, weed_density=18, output_path="test_data/field_sample.jpg")
    # Generate another with higher weed density
    generate_field_image(800, 600, num_rows=3, weed_density=35, output_path="test_data/field_sample_weedy.jpg")
    # Generate a clean field with few weeds
    generate_field_image(800, 600, num_rows=3, weed_density=5, output_path="test_data/field_sample_clean.jpg")
