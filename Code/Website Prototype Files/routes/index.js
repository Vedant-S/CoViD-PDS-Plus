//IMPORTING THE PACKAGES WE HAVE INSTALLED INTO OUR PROJECT
const express = require('express');
const router = express.Router();
const bodyParser = require('body-parser');


// Express body parser - MIDDLEWARE
router.use(bodyParser.json())
router.use(bodyParser.urlencoded({
    extended: true
}));


//GET REQUEST ON HOMEPAGE -> CALLING LIST.EJS
router.get('/',async (req, res) => {   
  res.render('index');
});

router.get('/model',async (req, res) => {   
  res.render('model');
});



//For exporting these settings. VERY IMPORTANT LINE. PEOPLE TEND TO MISS IT. Program wont run without this line.
module.exports = router;