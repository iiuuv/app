

def scale_zoom(width,height,scale,x,y):
    dx = x - width/2
    dy = y - height/2
    dx *= scale
    dy *= scale
    return dx +  width/2, dy + height/2


if __name__ == "__main__":
    x,y = scale_zoom(800, 800, 0.5, 100, 200)
    # Example usage
    # This will scale the zoom based on the center of the given width and height
    print(f"Scaled coordinates: ({x}, {y})")