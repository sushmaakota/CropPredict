const inputBox = document.querySelector('.input-box');
const searchBtn = document.getElementById('searchBtn');
const weather_img = document.querySelector('.weather-img');
const temperature = document.querySelector('.temperature');
const description = document.querySelector('.description');
const humidity = document.getElementById('humidity');
const wind_speed = document.getElementById('wind-speed');
const rainfall = document.getElementById('rainfall');

const location_not_found = document.querySelector('.location-not-found');

const weather_body = document.querySelector('.weather-body');


async function checkWeather(city){
    const api_key = "531c35bc4bb59b56fe54f3d73c80f6dc";
    const url = `https://api.openweathermap.org/data/2.5/weather?q=${city}&appid=${api_key}`;

    const weather_data = await fetch(`${url}`).then(response => response.json());


    if(weather_data.cod === `404`){
        location_not_found.style.display = "flex";
        weather_body.style.display = "none";
        console.log("error");
        return;
    }

    console.log("run");
    location_not_found.style.display = "none";
    weather_body.style.display = "flex";
    temperature.innerHTML = `${Math.round(weather_data.main.temp - 273.15)}°C`;
    description.innerHTML = `${weather_data.weather[0].description}`;

    humidity.innerHTML = `${weather_data.main.humidity}%`;
    wind_speed.innerHTML = `${weather_data.wind.speed}Km/H`;
    
    const rainVolume = weather_data.rain ? weather_data.rain["1h"] || weather_data.rain["3h"] || 0 : 0;
    rainfall.innerHTML = `${rainVolume} mm`;

    switch(weather_data.weather[0].main){
        case 'Clouds':
            weather_img.src = "img/cloud.png";
            break;
        case 'Clear':
            weather_img.src = "img/clear-sky.png";
            break;
        case 'Rain':
            weather_img.src = "img/rain.png";
            break;
        case 'Haze':
            weather_img.src = "img/haze.png";
            break;
        case 'Lightning':
            weather_img.src = "img/lightning.png";
            break;
        case 'Snow':
            weather_img.src = "img/snow.png";
            break;
        case 'Storm':
            weather_img.src = "img/storm.png";
            break;
        case 'Thunderstorm':
            weather_img.src = "img/thunderstorm.png";
            break;
        case 'Mist':
            weather_img.src = "img/mist.png";
            break;
        case 'Snow':
            weather_img.src = "img/snow.png";
            break;

    }

    console.log(weather_data);
}


searchBtn.addEventListener('click', ()=>{
    checkWeather(inputBox.value);
});