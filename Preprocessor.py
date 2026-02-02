import numpy as np
import cv2

# Target dimensions for the neural network (much more efficient)
TARGET_HEIGHT = 84
TARGET_WIDTH = 84

class Preprocessor:
    def __init__(self, screen_height, screen_width):
        self.screen_height = screen_height
        self.screen_width = screen_width
        self.target_height = TARGET_HEIGHT
        self.target_width = TARGET_WIDTH

    def preprocess(self, positions: np.ndarray) -> np.ndarray:
        """
        Turn the game state into a simple image (BGR format for OpenCV), then downscale.
        Enemies at their positions are red if going right, blue if going left.
        Player is green if bullet is ready, yellow if bullet is fired.

        Args:
            positions (np.ndarray): Current game state positions from environment (shape: (1, 1, 23)).

        Returns:
            Downscaled image of shape (84, 84, 3) normalized to [0, 1].
        """

        if isinstance(positions, np.ndarray):
            positions = positions.flatten()
        elif isinstance(positions, tuple):
            pass
        else:
            raise TypeError("Ummm. What's that?")

        image = np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8)

        # Unpack positions
        playerX, playerXD, bullet_ready, bulletX, bulletY = positions[:5]
        enemy_positions = positions[5:]

        # Draw player
        player_color = (0, 255, 0) if bullet_ready == 1 else (0, 255, 255)
        image[750:770, int(playerX):int(playerX)+50] = player_color

        # Draw bullet
        if bullet_ready == 0:
            image[int(bulletY):int(bulletY)+10, int(bulletX)+24:int(bulletX)+26] = (255, 255, 255)

        # Draw enemies
        for i in range(0, len(enemy_positions), 3):
            enemyX = enemy_positions[i]
            enemyY = enemy_positions[i+1]
            enemyXD = enemy_positions[i+2]
            enemy_color = (0, 0, 255) if enemyXD > 0 else (255, 0, 0)
            image[int(enemyY):int(enemyY)+40, int(enemyX):int(enemyX)+40] = enemy_color

        # Downscale image for neural network efficiency
        image = cv2.resize(image, (self.target_width, self.target_height), interpolation=cv2.INTER_AREA)
        
        # Normalize to [0, 1] for better neural network training
        image = image.astype(np.float32) / 255.0
        
        return image
    
if __name__ == "__main__":
    SCREEN_HEIGHT = 800
    SCREEN_WIDTH = 850
    preprocessor = Preprocessor(SCREEN_HEIGHT, SCREEN_WIDTH)
    dummy_positions = (400, 0, 1, 0, 0) + tuple([val for i in range(6) for val in ((100 + i*60, 50, 1) if i%2==0 else (100 + i*60, 50, -1))])
    processed_image = preprocessor.preprocess(dummy_positions)
    print(f"Output shape: {processed_image.shape}")  # Should output (84, 84, 3)
    print(f"Value range: [{processed_image.min():.2f}, {processed_image.max():.2f}]")  # Should be [0, 1]
    
    # Scale back up for display
    display_image = (processed_image * 255).astype(np.uint8)
    cv2.imshow("Preprocessed Image (84x84)", display_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()