import React from "react";
import "../style/loader.css";
import { Slider } from '@mui/material';

const CustomSlider = ({ defaultValue, setValue, min, max, step }) => {
    return (
        <Slider
            size="small"
            defaultValue={defaultValue}
            onChange={(e) => setValue(e)}
            aria-label="Default"
            valueLabelDisplay="auto"
            min={min}
            max={max}
            step={step}
        />
    );
};

export default CustomSlider;