#![allow(dead_code)]
#![allow(unused_imports)]
use std::io::Write;
use crate::types;
use crate::types::Color;

pub struct Canvas {
    pub width: usize,
    pub height: usize,
    pixels: Vec<types::Color>
}

impl Canvas {
    pub fn new(width: usize, height: usize) -> Canvas {
        let size = width * height;
        let pixels = vec![types::color(0.0, 0.0,0.0); size]; // assumes Color implements Default
        Self {
            width,
            height,
            pixels
        }
    }
}


pub fn write_pixel(canvas: &mut Canvas, x: usize, y: usize, color: types::Color) {
    if x < canvas.width && y < canvas.height {
        let idx = y * canvas.width + x;
        canvas.pixels[idx] = color;
    }
}

pub fn pixel_at(canvas: &Canvas, x: usize, y: usize) -> Result<types::Color, &'static str> {
    if x < canvas.width && y < canvas.height {
        let idx = y * canvas.width + x;
        Ok(canvas.pixels[idx])
    } else {
        Err("Pixel out of bounds")
    }
}

pub fn canvas_to_ppm_header(canvas: &Canvas) -> std::string::String {
    format!("P3\n{} {}\n255\n", canvas.width, canvas.height)
}

pub fn canvas_to_ppm_body(canvas: &Canvas) -> String {
    let mut ppm_string = String::new();
    let mut current_line_length = 0;
    const MAX_LINE_LENGTH: usize = 70;

    for y in 0..canvas.height {
        for x in 0..canvas.width {
            let pixel = &canvas.pixels[y * canvas.width + x];

            // Convert pixel values to 0-255 range and clamp
            let red = (pixel.red * 255.0).round().clamp(0.0, 255.0) as u8;
            let green = (pixel.green * 255.0).round().clamp(0.0, 255.0) as u8;
            let blue = (pixel.blue * 255.0).round().clamp(0.0, 255.0) as u8;

            // Process each color component
            for &component in &[red, green, blue] {
                let component_str = component.to_string();
                let space_needed = if current_line_length == 0 { 0 } else { 1 }; // space before number
                let total_needed = space_needed + component_str.len();

                // Check if adding this component would exceed the line limit
                if current_line_length > 0 && current_line_length + total_needed > MAX_LINE_LENGTH {
                    ppm_string.push('\n');
                    current_line_length = 0;
                }

                // Add space before component (except at start of line)
                if current_line_length > 0 {
                    ppm_string.push(' ');
                    current_line_length += 1;
                }

                // Add the component value
                ppm_string.push_str(&component_str);
                current_line_length += component_str.len();
            }
        }

        // End each row with a newline
        ppm_string.push('\n');
        current_line_length = 0;
    }

    ppm_string
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn canvas_creation() {
        let _canvas = Canvas::new(10, 20);
        assert_eq!(_canvas.width, 10);
        assert_eq!(_canvas.height, 20);
        for &pixel in _canvas.pixels.iter()
        {
            assert_eq!(pixel, types::color(0.0, 0.0, 0.0));
        }
    }

    #[test]
    fn write_pixel_to_canvas() {
        let mut canvas = Canvas::new(10, 20);
        let red = types::color(1.0, 0.0, 0.0);
        write_pixel(&mut canvas,2, 3, red);
        assert_eq!(pixel_at(&canvas, 2, 3).unwrap(), red);
    }

    #[test]
    fn canvas_to_ppm_header_test() {
        let canvas = Canvas::new(5, 3);
        let ppm_string = canvas_to_ppm_header(&canvas);
        assert_eq!(ppm_string, "P3\n5 3\n255\n");
    }
    #[test]
    fn canvas_to_ppm_body_test() {
        let mut canvas = Canvas::new(5, 3);
        let color1 = types::color(1.5, 0.0, 0.0);
        let color2 = types::color(0.0, 0.5, 0.0);
        let color3 = types::color(-0.5, 0.0, 1.0);
        write_pixel(&mut canvas, 0, 0, color1);
        write_pixel(&mut canvas, 2, 1, color2);
        write_pixel(&mut canvas, 4, 2, color3);
        let ppm_string = canvas_to_ppm_body(&canvas);
        assert_eq!(ppm_string,
                   "255 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n\
                   0 0 0 0 0 0 0 128 0 0 0 0 0 0 0\n\
                   0 0 0 0 0 0 0 0 0 0 0 0 0 0 255\n"
        )
    }
    #[test]
    fn canvas_to_ppm_body_max_line_len_test() {
        let mut canvas = Canvas::new(10, 2);
        let color = types::color(1.0, 0.8, 0.6);
        for x in 0..canvas.width {
            for y in 0..canvas.height {
                write_pixel(&mut canvas, x, y, color);
            }
        }
        let ppm_string = canvas_to_ppm_body(&canvas);
        assert_eq!(ppm_string,
                   "255 204 153 255 204 153 255 204 153 255 204 153 255 204 153 255 204\n\
                    153 255 204 153 255 204 153 255 204 153 255 204 153\n\
                    255 204 153 255 204 153 255 204 153 255 204 153 255 204 153 255 204\n\
                    153 255 204 153 255 204 153 255 204 153 255 204 153\n"
        )

    }
    #[test]
    fn canvas_to_ppm_body_new_line_test() {
        let canvas = Canvas::new(5, 3);
        let ppm_string = canvas_to_ppm_body(&canvas);
        assert!(ppm_string.ends_with('\n'));
    }

}