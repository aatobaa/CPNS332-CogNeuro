function [ direction ] = rad2direction( angle )
direction = [];
numAngles = size(angle);
x = 0:pi/4:2*pi;
for i = 1:numAngles
    d = abs(x-angle(i));
    [m dir] = min(d);
    if dir == 9
        dir= 1;
    end
    direction = [direction;dir];
end
end

