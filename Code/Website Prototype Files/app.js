//IMPORTING THE PACKAGES WE HAVE INSTALLED INTO OUR PROJECT
const express = require('express');
const path = require('path');
const bodyParser = require('body-parser');

//CREATING AN INSTANCE OF THE EXPRESS PACKAGE
const app = new express();

//SETTING THE PORT ON WHICH THE SERVER WILL RUN TO VARIABLE, SO THAT THE SYSTEM IN WHICH IT IS RUN DECIDES IT.
const PORT = process.env.PORT || 5000;

//Setting up the public directory => CONTAINS THE FILES THAT WILL BE ACCESSIBLE TO ALL.
//CONTAINS ALL MEDIA AND FILES REQUIRED IN THE WEB APP.
app.use(express.static('public'));

//Setting up the VIEWS directory => IN WHICH THE MAIN FRONTEND FILES ARE STORED.
app.set('views', path.join(__dirname, 'views'));


//SETTING VIEW TEMPLATE ENGINE
/*
      WE ARE USING EJS ENGINE FOR OUR FRONTEND. SO OUR EXTENSION FOR ALL HTML FILES WOULD BE .EJS. EJS HELPS US TO MAKE OUR PAGES MORE DYNAMIC.

      THERE ARE OTHER ENGINES LIKE (HBS), (EDGE), ETC. YOU CAN ALSO USE THEM.
*/
app.set('view engine', 'ejs');

// Express body parser - MIDDLEWARE
app.use(bodyParser.json())
app.use(bodyParser.urlencoded({ extended: true }));

//MAIN Routing FILE -> CONTAINS ALL THE CRUD OPERATIONS
app.use('/', require('./routes/index.js'));


/*    
      MAKE THE SYSTEM LISTEN TO THE PARTICULAR PORT DEFINED ABOVE.
      WE HAVE USED TEMPLATE LITERAL SYNTAX TO DYNAMICALLY CHANGE THE PORT NUMBER.
 */
app.listen(PORT, console.log(`Server started on port ${PORT}`));
