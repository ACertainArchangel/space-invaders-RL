import numpy as np
import cv2

class Preprocessor:
    def __init__(self, screen_height, screen_width):
        self.screen_height = screen_height
        self.screen_width = screen_width

    def preprocess(self, positions: np.ndarray) -> np.ndarray:
        """
        Turn the game state into a simple image (BGR format for OpenCV). Enemies at their positions are red if going right, blue if going left. Player is green if bullet is ready, yellow if bullet is fired.

        Args:
            positions (np.ndarray): Current game state positions from environment (shape: (1, 1, 23)).

        [self.playerX, self.playerXD, 1 if self.bullet_state == "ready" else 0, self.bulletX if self.bullet_state=="fire" else 0, 
        self.bulletY if self.bullet_state =="fire" else 0] +
        [value for e in self.enemies for value in (e.enemyX, e.enemyY, e.enemyXD)]
        """

        if isinstance(positions, np.ndarray):
            # Flatten the positions array if needed
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

        return image
    
if __name__ == "__main__":
    SCREEN_HEIGHT = 800
    SCREEN_WIDTH = 850
    Preprocessor = Preprocessor(SCREEN_HEIGHT, SCREEN_WIDTH)
    dummy_positions = (400, 0, 1, 0, 0) + tuple([val for i in range(6) for val in ((100 + i*60, 50, 1) if i%2==0 else (100 + i*60, 50, -1))])
    processed_image = Preprocessor.preprocess(dummy_positions)
    print(processed_image.shape)  # Should output (800, 850, 3)
    cv2.imshow("Preprocessed Image", processed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()