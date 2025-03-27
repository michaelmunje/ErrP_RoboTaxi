import pygame

pygame.init()
pygame.joystick.init()

# Initialize the first available Xbox controller
if pygame.joystick.get_count() > 0:
    controller = pygame.joystick.Joystick(0)
    controller.init()
    print(f"Detected controller: {controller.get_name()}")
else:
    print("No joystick detected!")
    pygame.quit()
    exit()

# Main event loop
running = True

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        
        # Detect joystick events
        if event.type == pygame.JOYAXISMOTION:
            print("Axis Motion Event Detected!")
            print(f"Joystick ID: {event.joy}, Axis Index: {event.axis}, Axis Value: {event.value:.2f}")

        elif event.type == pygame.JOYBUTTONDOWN:
            print(f"Button Pressed! Joystick ID: {event.joy}, Button Index: {event.button}")

        elif event.type == pygame.JOYBUTTONUP:
            print(f"Button Released! Joystick ID: {event.joy}, Button Index: {event.button}")

        elif event.type == pygame.JOYHATMOTION:
            print(f"D-Pad Moved! Joystick ID: {event.joy}, Hat Index: {event.hat}, Position: {event.value}")

        elif event.type == pygame.JOYDEVICEADDED:
            print(f"Joystick Connected! Joystick ID: {event.device_index}")

        elif event.type == pygame.JOYDEVICEREMOVED:
            print(f"Joystick Disconnected! Joystick ID: {event.joy}")

        print("-" * 30)

pygame.quit()
