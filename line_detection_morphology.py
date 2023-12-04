def morphology_based_line_detection(self, img):
        """
        detects lines of the img using morphological operators.
        :param img: the input image
        :return: an image with highlighted lines.
        """
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Apply adaptiveThreshold at the bitwise_not of gray
        bw = cv2.adaptiveThreshold(cv2.bitwise_not(img), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)
        horizontal = bw
        vertical = bw
        rows, cols = horizontal.shape
        horizontalsize = int(round(cols / 30))
        # Create structure element for extracting horizontal lines through morphology operations
        horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontalsize, 1))
        # Apply morphology operations
        horizontal = cv2.erode(horizontal, horizontalStructure, (-1, -1))
        horizontal = cv2.dilate(horizontal, horizontalStructure, (-1, -1))
        # Specify size on vertical axis
        verticalsize = int(round(rows / 30))
        # Create structure element for extracting vertical lines through morphology operations
        verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, verticalsize))
        # Apply morphology operations
        vertical = cv2.erode(vertical, verticalStructure, (-1, -1))
        vertical = cv2.dilate(vertical, verticalStructure, (-1, -1))
        line_img = cv2.bitwise_or(horizontal, vertical)
        return line_img
