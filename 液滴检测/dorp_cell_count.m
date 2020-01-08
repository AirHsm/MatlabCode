close all
clear all
clc

%img = ReadTIFF('./20200107-caski/2b.tif'); 	% bright field image
%fimg = ReadTIFF('./20200107-caski/3f.tif'); 	% fluorescent imasge 

img = ReadTIFF('5x-44.tif'); 	% bright field image
fimg = ReadTIFF('5x-44f.tif'); 	% fluorescent imasge 

img = img/max(img(:));          % intensity normalization
img = medfilt2(img);            % denoising


%% find cell
%{
thr = 0.35;

cell_mask = im2bw(fimg, thr);

se=strel('disk',5);
%cell_mask = imclose(cell_mask, se);
cell_mask = imopen(cell_mask, se);

[L,cell_num] = bwlabel(cell_mask);
cell = regionprops(L, 'Centroid');
cell_centers = zeros(cell_num,2);
for i = 1:cell_num
    cell_centers(i,:) = cell(i).Centroid;   % keep cell's center location in to an array.
end

display(['Cell number:',int2str(cell_num)]);

figure,
subplot(121), imshow(fimg);
subplot(122), imshow(cell_mask);

%}

%% find drops location from bf image
I = img;
[CENTERS, RADII, ~] = imfindcircles(I, [5,12], 'ObjectPolarity', 'bright');
figure,imshow(I),viscircles(CENTERS, RADII, 'EdgeColor', 'b');

[m,n] = size(I);
x = linspace(1,n,n);
y = linspace(1,m,m);
[X, Y] = meshgrid(x, y);

t = 1;
delta = 0;
cellnummax = 0;
mask = zeros(m,n);
RR = min(RADII);

for i = 1: max(size(CENTERS))
	if CENTERS(i,1) - RADII(i) > 1 && CENTERS(i,1) + RADII(i) < n && CENTERS(i,2) - RADII(i) > 1 && CENTERS(i,2) + RADII(i) < m
		drop(t).cent = CENTERS(i,:);
		drop(t).r = RADII(i);
        
        Rv = abs((X-CENTERS(i,1)) + 1i * (Y-CENTERS(i,2)));
	    %mask(Rv<(RADII(i)-delta)) = 1;
        mask(Rv<RR) = 1;
	    temp = I;
	    %temp(Rv>=(RADII(i)-delta)) = 0;
        temp(Rv>=RR) = 0;
	    %density(t) = sum(temp(:))/(2*pi*RADII(i)^2);         % calculate the mean intensity of each drops
		drop(t).density = sum(temp(:));%/(2*pi*RADII(i)^2);
        
        %drop(t).cellnum = CellNumCheck(drop(t).cent, drop(t).r, cell_centers);
        drop(t).STD = std(fimg(Rv<RR));
        %if drop(t).cellnum > cellnummax
        %    cellnummax = drop(t).cellnum;
        %end
        
        t = t+1;
	end
end

drop_num = max(size(drop));
display(['drop number:',int2str(drop_num)]);
display(['max number of cells in one drop:',int2str(cellnummax)]);

figure,imshow(mask.*fimg);
WriteTIFF(array2cell(mask.*fimg),'filted_img.tif');

%calculate std

STD = zeros(drop_num,1);
for i = 1:drop_num
    STD(i) = drop(i).STD;
end

figure, histogram(STD);

%% seperation

% seperating drops with non cells according to the histogram of intensity
% std.
t0 = 1;
t1 = 1;
for i = 1:drop_num
    if drop(i).STD < 0.04
        DropWithoutCells(t0) = drop(i);
        DropWithoutCells(t0).cellnum = 0;
        t0 = t0 + 1;
    else
        DropWithCells(t1) = drop(i);
        t1 = t1+1;
    end
end

% seperating single cell and multiple cells
t1 = 1;
t2 = 1;

for i = 1:max(size(DropWithCells))
    cent = DropWithCells(i).cent;
    r = DropWithCells(i).r;
    ROI = fimg(fix(cent(2)-r):fix(cent(2)+r),fix(cent(1)-r):fix(cent(1)+r));
    BW = imbinarize(ROI,0.5);
    [L,num] = bwlabel(BW);
    
    if num > 1
        DropWithCells(i).cellnum = 2;
    else
        Laxis = regionprops(L,'MajorAxisLength','MinorAxisLength');
        if Laxis.MajorAxisLength > 1.5*Laxis.MinorAxisLength
            DropWithCells(i).cellnum = 2;
        else
            DropWithCells(i).cellnum = 1;
        end
    end
end

    



%% analysis
%{
Density_Non_Cell = {};
Density_Single_Cell = {};
Density_Multiple_Cell = {};

t0 = 1;
t1 = 1;
t2 = 1;

for i = 1:drop_num
    if drop(i).cellnum == 0
        Density_Non_Cell{t0} = drop(i).density;
        D0(t0) = drop(i);
        t0 = t0 + 1;
    elseif drop(i).cellnum == 1
            Density_Single_Cell{t1} = drop(i).density;
            D1(t1) = drop(i);
            t1 = t1 + 1;
    elseif drop(i).cellnum > 1
        Density_Multiple_Cell{t2} = drop(i).density;
        D2(t2) = drop(i);
        t2 = t2 + 1;
    end
end

d0 = cell2array1d(Density_Non_Cell);
d1 = cell2array1d(Density_Single_Cell);
d2 = cell2array1d(Density_Multiple_Cell);

figure,
subplot(131), histogram(d0), title('Fluorescence density of drops without cells') ;
subplot(132), histogram(d1), title('Fluorescence density of drops with single cell');
subplot(133), histogram(d2), title('Fluorescence density of drops with multiple cells');        



% Check
DD = D0;
cc = zeros(max(size(DD)),2);
rr = zeros(max(size(DD)),1);
for i = 1:max(size(DD))
    cc(i,:) = DD(i).cent;
    rr(i) = DD(i).r;
end
figure, imshow(fimg), viscircles(cc,rr,'EdgeColor','b');
%}






