var request = require('request');
var app       = require('express')();
var http      = require('http').Server(app);
var path      = require('path');
var fs        = require('fs');
var express     = require('express');
var bodyParser  = require('body-parser');
var validator   = require('validator');

/* Setting app properties */
app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: true }));
app.use(require('express').static(__dirname+'/public'));
app.get('/', function (req, res) {
    res.sendFile(__dirname + '/index.html');
});

/* Listener */
http.listen(process.env.PORT || 3000, function() {
    console.log('listening on port:3000');
});